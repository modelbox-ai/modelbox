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


#ifndef MODELBOX_FLOWUNIT_FACE_MOBILEV2_FLOWUNIT_CPU_H_
#define MODELBOX_FLOWUNIT_FACE_MOBILEV2_FLOWUNIT_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME = "face_mobilev2";
constexpr const char *FLOWUNIT_DESC = R"(A cpu face_mobilev2 flowunit)";

static const std::vector<std::string> MOBILE_UNIT_IN_NAME = {"fc_blob1"};
static const std::vector<std::string> MOBILE_UNIT_OUT_NAME = {"Out_1"};

typedef struct tagFaceExpressionScore {
  float surprise_score;
  float fear_score;
  float disgust_score;
  float happy_score;
  float sad_score;
  float angry_score;
  float neutral_score;
} FaceExpressionScore;

class FaceMobilev2FlowUnit : public modelbox::FlowUnit {
 public:
  FaceMobilev2FlowUnit();
  virtual ~FaceMobilev2FlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);
  modelbox::Status Close();
  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

 private:
  modelbox::Status GetInferLayers(
      std::shared_ptr<modelbox::DataContext> data_ctx,
      std::vector<std::shared_ptr<modelbox::BufferList>> &infer_layers);
};

#endif  // MODELBOX_FLOWUNIT_FACE_MOBILEV2_FLOWUNIT_CPU_H_