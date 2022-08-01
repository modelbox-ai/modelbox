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

#ifndef EXAMPLE_PLUGIN_PLUGIN_H_
#define EXAMPLE_PLUGIN_PLUGIN_H_

#include <string>
#include "modelbox/base/status.h"
#include "modelbox/server/job_manager.h"
#include "modelbox/server/plugin.h"

class ExamplePlugin : public modelbox::Plugin {
 public:
  ExamplePlugin() = default;
  ~ExamplePlugin() override = default;

  virtual bool Init(std::shared_ptr<modelbox::Configuration> config) override;
  virtual bool Start() override;
  virtual bool Stop() override;
};

#endif  // EXAMPLE_PLUGIN_PLUGIN_H_