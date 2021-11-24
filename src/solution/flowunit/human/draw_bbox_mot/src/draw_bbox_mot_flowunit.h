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


#ifndef MODELBOX_FLOWUNIT_DRAW_BBOX_MOT_FLOWUNIT_CPU_H_
#define MODELBOX_FLOWUNIT_DRAW_BBOX_MOT_FLOWUNIT_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <string>

#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME = "draw_bbox_mot";
constexpr const char *FLOWUNIT_DESC = "A draw_bbox flowunit for mot  on CPU";
constexpr const char *FLOWUNIT_TYPE = "cpu";

const int EMBEDDING_LENGTH = 512;

typedef enum { New = 0, Tracked = 1, Lost = 2, Removed = 3 } TRACK_STATUS;

typedef struct BBox {
  float x, y, w, h;
  int category;
  float score;
} BBox;

typedef struct tag_Person {
  float emb[EMBEDDING_LENGTH];
  BBox personBox;
} Person;

typedef struct tag_TrackResult {
  int mTrackId = 0;
  BBox mPersonbox;
} TrackResult;

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

class DrawBBoxMOTFlowUnit : public modelbox::FlowUnit {
 public:
  DrawBBoxMOTFlowUnit();
  ~DrawBBoxMOTFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close() { return modelbox::STATUS_OK; };
  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

 private:
  // config to generate different color for trackids
  int prime_one_ = 37;
  int prime_two_ = 17;
  int prime_three_ = 29;
  int gray_range_ = 255;
  int ratio_ = 3;

  // config to draw the bounding bbox
  int thickness_ = 4;
  int linetype_ = 4;
  int shift_ = 0;

  // config to puttext
  int fontScale_ = 8;
};

#endif  // MODELBOX_FLOWUNIT_DRAW_BBOX_MOT_FLOWUNIT_CPU_H_
