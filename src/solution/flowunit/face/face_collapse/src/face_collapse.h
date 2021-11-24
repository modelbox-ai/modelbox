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


#ifndef MODELBOX_FLOWUNIT_FACE_COLLAPSE_FLOWUNIT_CPU_H_
#define MODELBOX_FLOWUNIT_FACE_COLLAPSE_FLOWUNIT_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <algorithm>

#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME = "face_collapse";
constexpr const char *FLOWUNIT_DESC = "A face_collapse flowunit on CPU";
constexpr const char *FLOWUNIT_TYPE = "cpu";

static const std::vector<std::string> COLLAPSE_UNIT_IN_NAME = {"In_label"};
static const std::vector<std::string> COLLAPSE_UNIT_OUT_NAME = {"Out_label"};

typedef struct tagFaceExpressionScore {
  float surprise_score;
  float fear_score;
  float disgust_score;
  float happy_score;
  float sad_score;
  float angry_score;
  float neutral_score;
} FaceExpressionScore;

class FaceCollapseFlowUnit : public modelbox::FlowUnit {
 public:
  FaceCollapseFlowUnit();
  ~FaceCollapseFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);
  modelbox::Status Close();

  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);
};

#endif  // MODELBOX_FLOWUNIT_FACE_COLLAPSE_FLOWUNIT_CPU_H_
