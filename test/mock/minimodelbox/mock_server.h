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

#ifndef MODELBOX_MOCKSERVERH_
#define MODELBOX_MOCKSERVERH_

#include <modelbox/base/configuration.h>

#include <iostream>
#include <string>
#include <thread>

#include "../server.h"
#include "modelbox/server/http_helper.h"

namespace modelbox {

class MockServer {
 public:
  MockServer();
  virtual ~MockServer();

  static std::string GetTestGraphDir();
  std::string GetServerURL();

  Status Init(std::shared_ptr<Configuration> config);
  Status Start();
  void Stop();
  httplib::Response DoRequest(HttpRequest &request);

 protected:
  virtual void SetDefaultConfig(const std::shared_ptr<Configuration> &config);

 private:
  std::shared_ptr<Server> server_;
};

}  // namespace modelbox
#endif  // MODELBOX_MOCKSERVERH_
