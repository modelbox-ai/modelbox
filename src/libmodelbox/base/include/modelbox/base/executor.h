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

#ifndef MODELBOX_EXECUTOR_H_
#define MODELBOX_EXECUTOR_H_

#include <modelbox/base/thread_pool.h>

namespace modelbox {

/**
 * @brief Executor for flowunit
 */
class Executor {
 public:
  Executor();
  Executor(int thread_count);
  Executor(const Executor &) = delete;
  Executor &operator=(const Executor &) = delete;

  virtual ~Executor();

  template <typename func, typename... ts>
  auto Run(func &&fun, int32_t priority, ts &&...params)
      -> std::future<typename std::result_of<func(ts...)>::type> {
    return thread_pool_->Submit(fun, params...);
  }

 private:
  std::shared_ptr<ThreadPool> thread_pool_;
};

}  // namespace modelbox

#endif
