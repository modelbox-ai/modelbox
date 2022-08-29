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

#include "example.h"

modelbox::Status ExampleOutputBroker::Init(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status ExampleOutputBroker::Deinit() { return modelbox::STATUS_OK; }

std::shared_ptr<modelbox::OutputBrokerHandle> ExampleOutputBroker::Open(
    const std::shared_ptr<modelbox::Configuration> &session_config,
    const std::string &config) {
  auto handle = std::make_shared<modelbox::OutputBrokerHandle>();
  // Your code goes here
  return handle;
}

modelbox::Status ExampleOutputBroker::Write(
    const std::shared_ptr<modelbox::OutputBrokerHandle> &handle,
    const std::shared_ptr<modelbox::Buffer> &buffer) {
  return modelbox::STATUS_OK;
}

modelbox::Status ExampleOutputBroker::Sync(
    const std::shared_ptr<modelbox::OutputBrokerHandle> &handle) {
  return modelbox::STATUS_OK;
}

modelbox::Status ExampleOutputBroker::Close(
    const std::shared_ptr<modelbox::OutputBrokerHandle> &handle) {
  return modelbox::STATUS_OK;
}
