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


#ifndef MODELBOX_FLOWUNIT_CARCONDITIONFLOWUNIT_CPU_H_
#define MODELBOX_FLOWUNIT_CARCONDITIONFLOWUNIT_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <algorithm>
#include <opencv2/opencv.hpp>

#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME = "has_bbox_condition";
constexpr const char *FLOWUNIT_DESC = "A car_condition flowunit on CPU";
constexpr const char *FLOWUNIT_TYPE = "cpu";

class CarConditionFlowUnit : public modelbox::FlowUnit {
 public:
  CarConditionFlowUnit();
  ~CarConditionFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close() { return modelbox::STATUS_OK; };
  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);
};

#endif  // MODELBOX_FLOWUNIT_CARCONDITIONFLOWUNIT_CPU_H_