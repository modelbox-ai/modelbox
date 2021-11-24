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


#ifndef MODELBOX_FLOWUNIT_MATCHINGFLOWUNIT_CPU_H_
#define MODELBOX_FLOWUNIT_MATCHINGFLOWUNIT_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <algorithm>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME = "matching";
constexpr const char *FLOWUNIT_DESC = "A matching flowunit for mot on CPU";
constexpr const char *FLOWUNIT_TYPE = "cpu";

constexpr const char *SELF_TRACKEDSTRACKS = "self_trackedstracks";
constexpr const char *SELF_LOSTSTRACKS = "self_loststracks";
constexpr const char *SELF_REMOVEDSTRACKS = "self_removedstracks";
constexpr const char *FRAMEID = "frameid";
constexpr const char *TRACKIDCOUNT = "trackidcount";

const int RGB_CHANNELS = 3;
const int EMBEDDING_LENGTH = 512;

typedef enum { New = 0, Tracked = 1, Lost = 2, Removed = 3 } TRACK_STATUS;

typedef struct tag_BBox {
  float x, y, w, h;
  int category;
  float score;
} BBox;

typedef struct tag_STrack {
  int mTrackId = 0;
  BBox mPersonbox;
  bool mIsActivated = false;
  int mTrackState = New;
  std::vector<float> mCurrFeature;
  std::vector<std::vector<float>> mHistoryFeatures;
  int mFrameID = 0;
  tag_STrack() { mCurrFeature.resize(EMBEDDING_LENGTH); }
} STrack;

typedef struct tag_TrackResult {
  int mTrackId;
  BBox mPersonbox;
} TrackResult;

typedef struct tag_MatchResult {
  std::vector<std::pair<int, int>> mMatches;
  std::vector<int> mUnTracked;
  std::vector<int> mNoneDetection;
} MatchResult;

typedef struct tag_Person {
  float emb[EMBEDDING_LENGTH]{0};
  BBox personBox;
} Person;

class MatchingFlowUnit : public modelbox::FlowUnit {
 public:
  MatchingFlowUnit();
  ~MatchingFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close() { return modelbox::STATUS_OK; };
  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

  modelbox::Status DataPre(std::shared_ptr<modelbox::DataContext> data_ctx);

  modelbox::Status DataPost(std::shared_ptr<modelbox::DataContext> data_ctx) {
    return modelbox::STATUS_OK;
  };

  modelbox::Status DataGroupPre(std::shared_ptr<modelbox::DataContext> data_ctx) {
    return modelbox::STATUS_OK;
  };

  modelbox::Status DataGroupPost(std::shared_ptr<modelbox::DataContext> data_ctx) {
    return modelbox::STATUS_OK;
  };

 private:
  void DealMatchStracks(std::vector<std::pair<int, int>> &matches,
                        std::vector<STrack> &strack_pool,
                        std::vector<STrack> &detections,
                        std::shared_ptr<int> frameID,
                        std::vector<STrack> &activated_starcks,
                        std::vector<STrack> &refind_stracks);
  void DealUnTrackedStracks(std::vector<int> &unTracked,
                            std::vector<STrack> &strack_pool,
                            std::vector<STrack> &lost_stracks);
  void DealNoneDetectionStracks(std::vector<int> &noneDetection,
                                std::vector<STrack> &detections,
                                std::shared_ptr<int> frameID,
                                std::shared_ptr<int> trackIDCount,
                                std::vector<STrack> &activated_starcks);
  void DealSelfLostStracks(std::shared_ptr<std::vector<STrack>> selfLostStracks,
                           std::shared_ptr<int> frameID,
                           std::vector<STrack> &removed_stracks);
  void GetTrackResults(std::shared_ptr<std::vector<STrack>> selfTrackedStracks,
                       std::vector<TrackResult> &trackResults);
  float CosineSimilarity(const std::vector<float> &detection,
                         const std::vector<float> &track,
                         const float &detfeatureMold,
                         const float &trackfeatureMold);
  std::pair<std::vector<std::vector<float>>, std::vector<int>>
  HistoryEmbeddingDistance(const std::vector<STrack> &trackPool,
                           const std::vector<STrack> &detections);
  MatchResult HistoryTopAssignment(
      std::pair<std::vector<std::vector<float>>, std::vector<int>>
          distanceResult,
      float threshold, const std::vector<STrack> &trackPool);
  std::vector<std::pair<int, int>> HistoryTopMatch(
      std::vector<std::vector<float>> costmatrix, float threshold,
      std::vector<int> mask);
  std::pair<int, int> FindMatrixMin(std::vector<std::vector<float>> matrix);
  int BisectLeft(const std::vector<int> &mask, int value);
  std::vector<STrack> JointStracks(std::vector<tag_STrack> trackedStracks,
                                   std::vector<tag_STrack> selfLostStracks);
  std::vector<STrack> SubStracks(std::vector<STrack> stracksOne,
                                 std::vector<STrack> stracksTwo);
  float threshold_ = 0.6;
  int maxTimeLost_ = 30;
  STrack UpdateStrack(STrack track, STrack currStrack, int frameID);
  STrack ReActivateStrack(STrack track, STrack currStrack, int frameID);
  STrack ActivateStrack(STrack track, int frameID, int trackid);
};

#endif  // MODELBOX_FLOWUNIT_MATCHINGFLOWUNIT_CPU_H_
