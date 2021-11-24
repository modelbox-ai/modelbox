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


#ifndef MODELBOX_FLOWUNIT_EXPANDBBOXIMGFLOWUNIT_CPU_H_
#define MODELBOX_FLOWUNIT_EXPANDBBOXIMGFLOWUNIT_CPU_H_

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

constexpr const char *FLOWUNIT_NAME = "pedestrian_expand_bbox_img";
constexpr const char *FLOWUNIT_DESC = "A expand_bbox_img flowunit on CPU";
constexpr const char *FLOWUNIT_TYPE = "cpu";
const int RGB_CHANNELS = 3;

const int MODEL_INPUT_W = 800;
const int MODEL_INPUT_H = 480;

typedef struct BBox {
  float x, y, w, h;
  int category;
  float score;
} BBox;

class ExpandBBoxImgFlowUnit : public modelbox::FlowUnit {
 public:
  ExpandBBoxImgFlowUnit();
  ~ExpandBBoxImgFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close();

  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);
};

#endif  // MODELBOX_FLOWUNIT_EXPANDBBOXIMGFLOWUNIT_CPU_H_
