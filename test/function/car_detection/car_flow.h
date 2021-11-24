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


#ifndef MODELBOX_CAR_FLOW_TEST_H_
#define MODELBOX_CAR_FLOW_TEST_H_

#include <modelbox/base/log.h>
#include <modelbox/flow.h>

#include <fstream>

#include "mock_driver_ctl.h"

namespace modelbox {

class CarFlow {
 public:
  CarFlow();
  virtual ~CarFlow();

  Status Init(const std::string &graphFilePath);
  Status Build();
  void Run();
  void Wait(const uint64_t millisecond);
  void Clear();
  void Destroy();

  std::shared_ptr<MockDriverCtl> GetMockFlowCtl();

 private:
  std::shared_ptr<Flow> flow_;
  std::shared_ptr<MockDriverCtl> ctl_;
};

}  // namespace modelbox

#endif  // MODELBOX_DRIVER_TEST_H_
