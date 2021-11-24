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


#ifndef MODELBOX_FLOWUNIT_PRE_PROCESS_FLOWUNIT_CPU_H_
#define MODELBOX_FLOWUNIT_PRE_PROCESS_FLOWUNIT_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <algorithm>
#include <opencv2/opencv.hpp>

#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME = "face_preprocess";
constexpr const char *FLOWUNIT_DESC = "A face_preprocess flowunit on CPU";
constexpr const char *FLOWUNIT_TYPE = "cpu";
static const int RGB_CHANNELS = 3;

static const std::vector<std::string> PREPROCESS_UNIT_IN_NAME = {"In_1"};
static const std::vector<std::string> PREPROCESS_UNIT_OUT_NAME = {"Out_1"};

class PreProcessFlowUnit : public modelbox::FlowUnit {
 public:
  PreProcessFlowUnit();
  virtual ~PreProcessFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);
  modelbox::Status Close();
  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

 private:
  cv::InterpolationFlags GetCVResizeMethod(std::string resizeType);

 private:
  uint32_t dest_width_{224};
  uint32_t dest_height_{224};
  std::string method_{"INTER_LINEAR"};
  std::vector<double> normalizes_{0.003921568627451, 0.003921568627451,
                                  0.003921568627451};
  std::vector<double> localmeans_{0.408, 0.447, 0.470};
  std::vector<double> variances_{0.289, 0.274, 0.278};
};

#endif  // MODELBOX_FLOWUNIT_PRE_PROCESS_FLOWUNIT_CPU_H_
