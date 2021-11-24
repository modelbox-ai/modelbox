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


#ifndef MODELBOX_FLOWUNIT_OUTPUT_BROKER_PLUGIN_H_
#define MODELBOX_FLOWUNIT_OUTPUT_BROKER_PLUGIN_H_

#include <modelbox/base/driver.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>

#include <memory>
#include <string>

constexpr const char *DRIVER_CLASS_OUTPUT_BROKER_PLUGIN =
    "DRIVER-OUTPUT-BROKER";

class OutputBrokerHandle {
 public:
  std::string output_broker_type_;
  std::string broker_id_;
};

class OutputBrokerPlugin : public modelbox::Driver {
 public:
  virtual modelbox::Status Init(
      const std::shared_ptr<modelbox::Configuration> &opts) = 0;

  virtual modelbox::Status Deinit() = 0;

  virtual std::shared_ptr<OutputBrokerHandle> Open(
      const std::string &config) = 0;

  virtual modelbox::Status Write(
      const std::shared_ptr<OutputBrokerHandle> &handle,
      const std::shared_ptr<modelbox::Buffer> &buffer) = 0;

  virtual modelbox::Status Sync(
      const std::shared_ptr<OutputBrokerHandle> &handle) = 0;

  virtual modelbox::Status Close(
      const std::shared_ptr<OutputBrokerHandle> &handle) = 0;
};

#endif  // MODELBOX_FLOWUNIT_OUTPUT_BROKER_PLUGIN_H_