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

#ifndef MODELBOX_SERVER_H_
#define MODELBOX_SERVER_H_

#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>

#include <iostream>
#include <memory>
#include <utility>

#include "control.h"
#include "server_plugin.h"

namespace modelbox {

class Server {
 public:
  Server(std::shared_ptr<modelbox::Configuration> config)
      : config_(std::move(config)){};
  virtual ~Server();
  modelbox::Status Init();
  modelbox::Status Start();
  modelbox::Status Stop();
  modelbox::Status Check();

 private:
  modelbox::Status GetPluginList();

  std::vector<std::shared_ptr<ServerPlugin>> plugins_;
  std::shared_ptr<modelbox::Configuration> config_;
  Control control_;
};

}  // namespace modelbox

#endif  // MODELBOX_SERVER_H_
