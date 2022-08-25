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

#ifndef MODELBOX_FLOWUNIT_CPU_PASSTHROUGH_H_
#define MODELBOX_FLOWUNIT_CPU_PASSTHROUGH_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/device/cpu/device_cpu.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_NAME = "passthrouth";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: A passthrouth flowunit on cpu device. \n";

class PassThrouthFlowUnit : public modelbox::FlowUnit {
 public:
  PassThrouthFlowUnit() = default;
  ~PassThrouthFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override;

  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;
};

#endif  // MODELBOX_FLOWUNIT_CPU_PASSTHROUGH_H_
