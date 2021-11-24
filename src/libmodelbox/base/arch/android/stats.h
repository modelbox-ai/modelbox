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


#include <iostream>
#include "modelbox/base/os.h"

namespace modelbox {

class AndroidOSProcess : public OSProcess {
 public:
  AndroidOSProcess();
  virtual ~AndroidOSProcess();

  virtual int32_t GetThreadsNumber(uint32_t pid);
  virtual uint32_t GetMemorySize(uint32_t pid);
  virtual uint32_t GetMemoryRSS(uint32_t pid);
  virtual uint32_t GetMemorySHR(uint32_t pid);
  virtual uint32_t GetPid();

  virtual std::vector<uint32_t> GetProcessTime(uint32_t pid);
  virtual std::vector<uint32_t> GetTotalTime(uint32_t pid);
};

}  // namespace modelbox