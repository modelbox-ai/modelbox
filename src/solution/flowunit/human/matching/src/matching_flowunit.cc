/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "matching_flowunit.h"

#include <securec.h>
#include <time.h>

#include <algorithm>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

#include <assert.h>
#ifdef DEBUG
#define ASSERT(f) assert(f)
#else
#define ASSERT(f) ((void)0)
#endif

MatchingFlowUnit::MatchingFlowUnit(){};
MatchingFlowUnit::~MatchingFlowUnit(){};

modelbox::Status MatchingFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}
modelbox::Status MatchingFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  MBLOG_DEBUG << "matching pedestrian on cpu";

  auto selfTrackedStracks = std::static_pointer_cast<std::vector<STrack>>(
      ctx->GetPrivate(SELF_TRACKEDSTRACKS));
  auto selfLostStracks = std::static_pointer_cast<std::vector<STrack>>(
      ctx->GetPrivate(SELF_LOSTSTRACKS));
  auto selfRemovedStracks = std::static_pointer_cast<std::vector<STrack>>(
      ctx->GetPrivate(SELF_REMOVEDSTRACKS));
  auto frameID = std::static_pointer_cast<int>(ctx->GetPrivate(FRAMEID));
  auto trackIDCount =
      std::static_pointer_cast<int>(ctx->GetPrivate(TRACKIDCOUNT));

  auto input1_bufs = ctx->Input("Input_1");
  auto output_bufs = ctx->Output("Output");

  std::vector<size_t> shape;
  std::vector<std::vector<TrackResult>> trackResultsVector;

  for (size_t i = 0; i < input1_bufs->Size(); ++i) {
    size_t num_persons = input1_bufs->At(i)->GetBytes() / sizeof(Person);

    MBLOG_DEBUG << "num_persons: " << num_persons;

    std::vector<STrack> detections;
    for (size_t j = 0; j < num_persons; ++j) {
      Person person;
      auto input_ret =
          memcpy_s(&person, sizeof(Person),
                   (const char *)(input1_bufs->ConstBufferData(i)) +
                       (sizeof(Person) * j),
                   sizeof(Person));
      if (EOK != input_ret) {
        MBLOG_ERROR << "cpu matching failed, input_ret " << input_ret;
        return modelbox::STATUS_FAULT;
      }

      STrack detection;
      std::vector<float> embedding(person.emb, person.emb + EMBEDDING_LENGTH);
      detection.mCurrFeature = embedding;
      detection.mHistoryFeatures.push_back(detection.mCurrFeature);
      detection.mPersonbox = person.personBox;
      detections.push_back(detection);
    }

    std::vector<STrack> trackedstracks;
    std::vector<STrack> strack_pool;
    std::vector<STrack> activated_starcks;
    std::vector<STrack> refind_stracks;
    std::vector<STrack> lost_stracks;
    std::vector<STrack> removed_stracks;

    for (auto &it : *selfTrackedStracks) {
      if (it.mIsActivated) {
        trackedstracks.push_back(it);
      }
    }

    strack_pool = JointStracks(trackedstracks, *selfLostStracks);

    std::pair<std::vector<std::vector<float>>, std::vector<int>> distanceResult;
    distanceResult = HistoryEmbeddingDistance(strack_pool, detections);

    MatchResult matchResult;
    matchResult = HistoryTopAssignment(distanceResult, threshold_, strack_pool);

    std::vector<std::pair<int, int>> matches;
    std::vector<int> unTracked;
    std::vector<int> noneDetection;

    matches = matchResult.mMatches;
    unTracked = matchResult.mUnTracked;
    noneDetection = matchResult.mNoneDetection;

    DealMatchStracks(matches, strack_pool, detections, frameID,
                     activated_starcks, refind_stracks);
    DealUnTrackedStracks(unTracked, strack_pool, lost_stracks);
    DealNoneDetectionStracks(noneDetection, detections, frameID, trackIDCount,
                             activated_starcks);
    DealSelfLostStracks(selfLostStracks, frameID, removed_stracks);

    std::vector<STrack> selfSaveTrackedStracks;
    std::vector<STrack> selfSaveLostStracks;
    std::vector<STrack> selfSaveRomovedStracks;
    std::vector<STrack> OutputStracks;

    selfTrackedStracks->erase(selfTrackedStracks->begin(),
                              selfTrackedStracks->end());

    for (auto &it : activated_starcks) {
      if (it.mTrackState == Tracked) {
        selfTrackedStracks->push_back(it);
      }
    }

    std::vector<STrack> ViewStracks = *selfTrackedStracks;
    selfSaveTrackedStracks =
        JointStracks(selfSaveTrackedStracks, refind_stracks);
    selfSaveLostStracks = SubStracks(*selfLostStracks, selfSaveTrackedStracks);
    selfSaveLostStracks.insert(selfSaveLostStracks.end(), lost_stracks.begin(),
                               lost_stracks.end());
    selfSaveLostStracks = SubStracks(selfSaveLostStracks, *selfRemovedStracks);
    selfSaveRomovedStracks.insert(selfSaveRomovedStracks.end(),
                                  removed_stracks.begin(),
                                  removed_stracks.end());

    *selfTrackedStracks = JointStracks(*selfTrackedStracks, refind_stracks);
    *selfLostStracks = SubStracks(*selfLostStracks, *selfTrackedStracks);
    selfLostStracks->insert(selfLostStracks->end(), lost_stracks.begin(),
                            lost_stracks.end());
    *selfLostStracks = SubStracks(*selfLostStracks, *selfRemovedStracks);
    *selfRemovedStracks->insert(selfRemovedStracks->end(),
                                removed_stracks.begin(), removed_stracks.end());

    std::vector<TrackResult> trackResults;
    GetTrackResults(selfTrackedStracks, trackResults);

    size_t data_size = sizeof(TrackResult);
    shape.push_back(trackResults.size() * data_size);
    trackResultsVector.push_back(trackResults);

    (*frameID)++;
  }

  auto shape_ret = output_bufs->Build(shape);
  if (!shape_ret) {
    MBLOG_ERROR << "matching : get output memory failed.";
    return modelbox::STATUS_NOMEM;
  }

  size_t data_size = sizeof(TrackResult);
  for (size_t i = 0; i < input1_bufs->Size(); ++i) {
    auto output_data = (TrackResult *)(output_bufs->MutableBufferData(i));
    for (auto &output : trackResultsVector[i]) {
      auto output_ret = memcpy_s(output_data, data_size, &output, data_size);
      if (EOK != output_ret) {
        MBLOG_ERROR << "Cpu matching failed, output_ret " << output_ret;
        return modelbox::STATUS_FAULT;
      }
      output_data++;
    }
  }

  ctx->SetPrivate(SELF_TRACKEDSTRACKS, selfTrackedStracks);
  ctx->SetPrivate(SELF_LOSTSTRACKS, selfLostStracks);
  ctx->SetPrivate(SELF_REMOVEDSTRACKS, selfRemovedStracks);
  ctx->SetPrivate(FRAMEID, frameID);
  ctx->SetPrivate(TRACKIDCOUNT, trackIDCount);

  MBLOG_DEBUG << "Matching finish";
  return modelbox::STATUS_OK;
}

void MatchingFlowUnit::DealMatchStracks(
    std::vector<std::pair<int, int>> &matches, std::vector<STrack> &strack_pool,
    std::vector<STrack> &detections, std::shared_ptr<int> frameID,
    std::vector<STrack> &activated_starcks,
    std::vector<STrack> &refind_stracks) {
  for (auto &it : matches) {
    int itracked = it.first;
    int idet = it.second;
    STrack track = strack_pool[itracked];
    STrack currtrack = detections[idet];
    if (track.mTrackState == Tracked) {
      STrack newTrack = UpdateStrack(track, currtrack, *frameID);
      activated_starcks.push_back(newTrack);
    } else if (track.mTrackState == Lost) {
      STrack reTrack = ReActivateStrack(track, currtrack, *frameID);
      refind_stracks.push_back(reTrack);
    }
  }
}

void MatchingFlowUnit::DealUnTrackedStracks(std::vector<int> &unTracked,
                                            std::vector<STrack> &strack_pool,
                                            std::vector<STrack> &lost_stracks) {
  for (auto &it : unTracked) {
    STrack track = strack_pool[it];
    if (track.mTrackState != Lost) {
      track.mTrackState = Lost;
      lost_stracks.push_back(track);
    }
  }
}

void MatchingFlowUnit::DealNoneDetectionStracks(
    std::vector<int> &noneDetection, std::vector<STrack> &detections,
    std::shared_ptr<int> frameID, std::shared_ptr<int> trackIDCount,
    std::vector<STrack> &activated_starcks) {
  for (auto &it : noneDetection) {
    STrack track = detections[it];
    if (track.mPersonbox.score < 0.4) {
      continue;
    }

    *trackIDCount += 1;
    STrack acTrack = ActivateStrack(track, *frameID, *trackIDCount);
    activated_starcks.push_back(acTrack);
  }
}

void MatchingFlowUnit::DealSelfLostStracks(
    std::shared_ptr<std::vector<STrack>> selfLostStracks,
    std::shared_ptr<int> frameID, std::vector<STrack> &removed_stracks) {
  for (auto &it : *selfLostStracks) {
    if ((*frameID - it.mFrameID) > maxTimeLost_) {
      it.mTrackState = Removed;
      removed_stracks.push_back(it);
    }
  }
}

void MatchingFlowUnit::GetTrackResults(
    std::shared_ptr<std::vector<STrack>> selfTrackedStracks,
    std::vector<TrackResult> &trackResults) {
  for (auto &it : *selfTrackedStracks) {
    if (it.mIsActivated) {
      TrackResult trackResult;
      trackResult.mTrackId = it.mTrackId;
      trackResult.mPersonbox = it.mPersonbox;
      trackResults.push_back(trackResult);
    }
  }
}

float MatchingFlowUnit::CosineSimilarity(const std::vector<float> &detection,
                                         const std::vector<float> &track,
                                         const float &detfeatureMold,
                                         const float &trackfeatureMold) {
  unsigned int n = detection.size();
  ASSERT(n == track.size());
  float tmp = 0.0;
  for (unsigned int i = 0; i < n; ++i) {
    tmp += track[i] * detection[i];
  }

  float multiplyMold = trackfeatureMold * detfeatureMold;

  if (0 == multiplyMold) {
    return 1;
  }

  return 1 - tmp / multiplyMold;
}

std::pair<std::vector<std::vector<float>>, std::vector<int>>
MatchingFlowUnit::HistoryEmbeddingDistance(
    const std::vector<STrack> &trackPool,
    const std::vector<STrack> &detections) {
  std::pair<std::vector<std::vector<float>>, std::vector<int>> result;

  std::vector<std::vector<float>> trackfeatures;
  std::vector<float> trackfeatureMolds;
  std::vector<int> mask;
  for (unsigned int j = 0; j < trackPool.size(); j++) {
    std::vector<std::vector<float>> HisFeatures = trackPool[j].mHistoryFeatures;
    int featurenums = HisFeatures.size();
    if (mask.empty()) {
      mask.push_back(featurenums - 1);
    } else {
      mask.push_back(featurenums + mask.back());
    }
    trackfeatures.insert(trackfeatures.end(), HisFeatures.begin(),
                         HisFeatures.end());
  }

  std::vector<std::vector<float>> costmatrix(
      detections.size(), std::vector<float>(trackfeatures.size(), 0.0));

  result.first = costmatrix;
  result.second = mask;

  std::vector<std::vector<float>> detfeatures;

  if (trackPool.empty() || detections.empty()) {
    return result;
  }

  for (unsigned int i = 0; i < detections.size(); i++) {
    detfeatures.push_back(detections[i].mCurrFeature);
  }

  cv::Mat detfeaturesMat(detfeatures.size(), detfeatures.at(0).size(), CV_32F);
  cv::Mat trackfeaturesMat(trackfeatures.size(), trackfeatures.at(0).size(),
                           CV_32F);
  cv::Mat costmatrixMat(detfeatures.size(), trackfeatures.size(), CV_32F);

  for (unsigned int k = 0; k < detfeatures.size(); ++k) {
    for (unsigned int o = 0; o < detfeatures.at(0).size(); ++o) {
      detfeaturesMat.at<float>(k, o) = detfeatures[k][o];
    }
  }

  for (unsigned int l = 0; l < trackfeatures.size(); ++l) {
    for (unsigned int p = 0; p < trackfeatures.at(0).size(); ++p) {
      trackfeaturesMat.at<float>(l, p) = trackfeatures[l][p];
    }
  }

  trackfeaturesMat = trackfeaturesMat.t();
  costmatrixMat = detfeaturesMat * trackfeaturesMat;

  for (unsigned int m = 0; m < detfeatures.size(); m++) {
    for (unsigned int n = 0; n < trackfeatures.size(); n++) {
      float cost = costmatrixMat.at<float>(m, n);
      costmatrix[m][n] = cost;
    }
  }

  unsigned int k, m, l;
  std::vector<float> detfeature;
  std::vector<float> trackfeature;
  float tmp;

  for (k = 0; k < detfeatures.size(); ++k) {
    for (l = 0; l < trackfeatures.size(); ++l) {
      tmp = 1.0;
      detfeature = detfeatures[k];
      trackfeature = trackfeatures[l];
      costmatrix[k][l] =
          CosineSimilarity(detfeatures[k], trackfeatures[l], 1, 1);
      for (m = 0; m < EMBEDDING_LENGTH; ++m) {
        tmp -= detfeature[m] * trackfeature[m];
      }
      costmatrix[k][l] = tmp;
    }
  }

  result.first = costmatrix;
  result.second = mask;
  return result;
}

MatchResult MatchingFlowUnit::HistoryTopAssignment(
    std::pair<std::vector<std::vector<float>>, std::vector<int>> distanceResult,
    float threshold, const std::vector<STrack> &trackPool) {
  MatchResult matchResult;
  std::vector<std::pair<int, int>> matches;
  std::vector<int> unTracked;
  std::vector<int> noneDetection;

  std::vector<std::vector<float>> costmatrix = distanceResult.first;
  std::vector<int> mask = distanceResult.second;

  for (unsigned int i = 0; i < trackPool.size(); i++) {
    unTracked.push_back(i);
  }
  for (unsigned int j = 0; j < costmatrix.size(); j++) {
    noneDetection.push_back(j);
  }

  matchResult.mMatches = matches;
  matchResult.mUnTracked = unTracked;
  matchResult.mNoneDetection = noneDetection;

  if (trackPool.empty()) {
    return matchResult;
  }
  matches = HistoryTopMatch(costmatrix, threshold, mask);
  for (auto match : matches) {
    for (auto it = unTracked.begin(); it != unTracked.end();) {
      if (*it == match.first)
        it = unTracked.erase(it);
      else
        ++it;
    }
    for (auto it = noneDetection.begin(); it != noneDetection.end();) {
      if (*it == match.second)
        it = noneDetection.erase(it);
      else
        ++it;
    }
  }
  matchResult.mMatches = matches;
  matchResult.mUnTracked = unTracked;
  matchResult.mNoneDetection = noneDetection;
  return matchResult;
}

std::vector<std::pair<int, int>> MatchingFlowUnit::HistoryTopMatch(
    std::vector<std::vector<float>> costmatrix, float threshold,
    std::vector<int> mask) {
  std::vector<std::pair<int, int>> matches;
  for (unsigned int i = 0; i < costmatrix.size(); i++) {
    std::pair<int, int> index = FindMatrixMin(costmatrix);
    float track_min = costmatrix[index.first][index.second];
    if (track_min < threshold) {
      for (unsigned int j = 0; j < costmatrix[index.first].size(); j++) {
        costmatrix[index.first][j] = 2.0;
      }
      int track_index = BisectLeft(mask, index.second);
      std::pair<int, int> match(track_index, index.first);
      matches.push_back(match);
      if (track_index == 0) {
        for (unsigned int k = 0; k < costmatrix.size(); k++) {
          for (int l = 0; l < mask[track_index] + 1; l++) {
            costmatrix[k][l] = 2.0;
          }
        }
      } else {
        for (unsigned int m = 0; m < costmatrix.size(); m++) {
          for (int n = mask[track_index - 1] + 1; n < mask[track_index] + 1;
               n++) {
            costmatrix[m][n] = 2.0;
          }
        }
      }
    }
  }
  return matches;
}

std::pair<int, int> MatchingFlowUnit::FindMatrixMin(
    std::vector<std::vector<float>> matrix) {
  float minValue = 2.0;
  std::pair<int, int> index(0, 0);
  for (unsigned int i = 0; i < matrix.size(); i++) {
    for (unsigned int j = 0; j < matrix[0].size(); j++) {
      if (matrix[i][j] < minValue) {
        index.first = i;
        index.second = j;
        minValue = matrix[i][j];
      }
    }
  };
  return index;
}

int MatchingFlowUnit::BisectLeft(const std::vector<int> &mask, int value) {
  int low = 0;
  int high = mask.size();
  while (low < high) {
    int mid = (low + high) / 2;
    if (mask[mid] < value) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

std::vector<STrack> MatchingFlowUnit::JointStracks(
    std::vector<STrack> trackedStracks, std::vector<STrack> selfLostStracks) {
  std::vector<STrack> result;
  std::map<int, int> exists;
  for (auto &it : trackedStracks) {
    exists[it.mTrackId] = 1;
    result.push_back(it);
  }
  for (auto &it : selfLostStracks) {
    int tid = it.mTrackId;
    if (exists.find(tid) == exists.end()) {
      exists[tid] = 1;
      result.push_back(it);
    }
  }
  return result;
}
std::vector<STrack> MatchingFlowUnit::SubStracks(
    std::vector<STrack> stracksOne, std::vector<STrack> stracksTwo) {
  std::map<int, STrack> stracks;
  std::vector<STrack> results;
  for (auto &it : stracksOne) {
    stracks[it.mTrackId] = it;
  }
  for (auto &it : stracksTwo) {
    int tid = it.mTrackId;

    std::map<int, STrack>::iterator key = stracks.find(tid);
    if (key != stracks.end()) {
      stracks.erase(key);
    }
  }
  for (auto &it : stracks) {
    results.push_back(it.second);
  }
  return results;
}

STrack MatchingFlowUnit::UpdateStrack(STrack track, STrack currStrack,
                                      int frameID) {
  STrack newtrack = track;
  newtrack.mFrameID = frameID;
  newtrack.mTrackState = Tracked;
  newtrack.mIsActivated = true;
  newtrack.mPersonbox = currStrack.mPersonbox;
  newtrack.mCurrFeature = currStrack.mCurrFeature;
  newtrack.mHistoryFeatures.push_back(currStrack.mCurrFeature);
  return newtrack;
}

STrack MatchingFlowUnit::ActivateStrack(STrack track, int frameID,
                                        int trackid) {
  STrack newtrack = track;
  newtrack.mTrackId = trackid;
  newtrack.mTrackState = Tracked;
  newtrack.mFrameID = frameID;
  newtrack.mIsActivated = true;
  return newtrack;
}

STrack MatchingFlowUnit::ReActivateStrack(STrack track, STrack currStrack,
                                          int frameID) {
  STrack retrack = track;
  retrack.mHistoryFeatures.push_back(currStrack.mCurrFeature);
  retrack.mCurrFeature = currStrack.mCurrFeature;
  retrack.mFrameID = frameID;
  retrack.mTrackState = Tracked;
  retrack.mIsActivated = true;
  retrack.mPersonbox = currStrack.mPersonbox;
  return retrack;
}

modelbox::Status MatchingFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto selfTrackedStracks = std::make_shared<std::vector<STrack>>();
  auto selfLostStracks = std::make_shared<std::vector<STrack>>();
  auto selfRemovedStracks = std::make_shared<std::vector<STrack>>();
  auto frameID = std::make_shared<int>();
  auto trackIDCount = std::make_shared<int>();

  *frameID = 0;
  *trackIDCount = 0;

  data_ctx->SetPrivate(SELF_TRACKEDSTRACKS, selfTrackedStracks);
  data_ctx->SetPrivate(SELF_LOSTSTRACKS, selfLostStracks);
  data_ctx->SetPrivate(SELF_REMOVEDSTRACKS, selfRemovedStracks);
  data_ctx->SetPrivate(FRAMEID, frameID);
  data_ctx->SetPrivate(TRACKIDCOUNT, trackIDCount);

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(MatchingFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("Input_1", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("Output", modelbox::DEVICE_TYPE));
  desc.SetFlowType(modelbox::STREAM);
  desc.SetStreamSameCount(true);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
