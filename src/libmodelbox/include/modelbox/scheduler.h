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


#ifndef MODELBOX_SCHEDULER_H_
#define MODELBOX_SCHEDULER_H_

#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>
#include <modelbox/base/thread_pool.h>
#include <modelbox/port.h>

#include <memory>
#include <string>
#include <vector>

namespace modelbox {

class Graph;

/**
 * @brief Scheduler run mode
 */
enum RunMode {
  /// run sync
  SYNC = 0,
  /// run async
  ASYNC = 1,
};

class Scheduler {
 public:
  virtual ~Scheduler() = default;

  /**
   * @brief Init scheduler
   * @param config scheduler configuration
   * @param stats scheduler stats
   * @param thread_pool thread pool for scheduler, if null, scheduler will
   * create its own thread pool
   * @return init result
   */
  virtual Status Init(std::shared_ptr<Configuration> config,
                      std::shared_ptr<StatisticsItem> stats = nullptr,
                      std::shared_ptr<ThreadPool> thread_pool = nullptr) = 0;


  /**
   * @brief Build graph
   * @param graph graph
   * @return build result
   */
  virtual Status Build(const Graph& graph) = 0;

  /**
   * @brief Run scheduler sync
   * @return run result
   */
  virtual Status Run() = 0;

  /**
   * @brief Run scheduler async
   */
  virtual void RunAsync() = 0;

  /**
   * @brief Wait for scheduler result
   * @param milliseconds timeout millisecond
   * @param ret_val graph result.
   * @return wait result
   */
  virtual Status Wait(int64_t milliseconds, Status* ret_val = nullptr) = 0;

  /**
   * @brief Shutdown scheduler
   */
  virtual void Shutdown() = 0;
};

}  // namespace modelbox

#endif
