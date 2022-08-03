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

#ifndef MODELBOX_FLOWUNIT_OUTPUT_BROKER_CPU_H_
#define MODELBOX_FLOWUNIT_OUTPUT_BROKER_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <algorithm>
#include <atomic>
#include <nlohmann/json.hpp>

#include "modelbox/base/timer.h"
#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"
#include "output_broker_plugin.h"

constexpr const char *FLOWUNIT_NAME = "output_broker";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Output the input data to the specified service. Currently "
    "supported types have dis, obs, webhook. \n"
    "\t@Port parameter: the input port buffer contain the following meta "
    "fields:\n"
    "\t\tField Name: out_broker_names,      Type: string\n"
    "\t\tField Name: out_file_names,        Type: string\n"
    "\t@Constraint: the fields 'out_file_names' can be only required when "
    "output "
    "type is obs. ";

constexpr const char *INPUT_DATA = "in_output_info";
constexpr const char *META_OUTPUT_BROKER_NAME = "output_broker_names";
constexpr const char *META_OUTPUT_FILE_NAME = "output_file_name";
constexpr const char *SESSION_OUTPUT_BROKER_CONFIG = "config";
constexpr const char *CTX_BROKER_NAMES = "broker_names";
constexpr const char *CTX_BROKER_INSTANCES = "broker_instances";
constexpr const char *SYNC_MODE = "sync";
constexpr const char *ASYNC_MODE = "async";

using BrokerNames = std::vector<std::string>;

class BrokerDataQueue {
 public:
  BrokerDataQueue(std::string broker_name, size_t queue_size);

  virtual ~BrokerDataQueue() = default;

  void PushForce(const std::shared_ptr<modelbox::Buffer> &buffer);

  modelbox::Status Front(std::shared_ptr<modelbox::Buffer> &buffer);

  bool Empty();

  void PopIfEqual(const std::shared_ptr<modelbox::Buffer> &target);

 private:
  std::string broker_name_;
  size_t queue_size_{0};
  std::queue<std::shared_ptr<modelbox::Buffer>> queue_;
  std::mutex queue_lock_;
};

class BrokerInstance {
 public:
  BrokerInstance(std::shared_ptr<OutputBrokerPlugin> &plugin,
                 const std::string &name,
                 std::shared_ptr<OutputBrokerHandle> &handle,
                 size_t async_queue_size);

  virtual ~BrokerInstance();

  void SetRetryParam(int64_t retry_count_limit, size_t retry_interval_base_ms,
                     size_t retry_interval_increment_ms,
                     size_t retry_interval_limit_ms);

  modelbox::Status Write(const std::shared_ptr<modelbox::Buffer> &buffer);

  modelbox::Status AddToQueue(const std::shared_ptr<modelbox::Buffer> &buffer);

  void WriteFromQueue();

  void Dispose();

  std::atomic_bool is_stopped{true};

 private:
  void UpdateInstaceState(modelbox::Status write_result);

  std::shared_ptr<OutputBrokerPlugin> plugin_;
  std::string name_;
  std::shared_ptr<OutputBrokerHandle> handle_;
  BrokerDataQueue data_queue_;

  size_t send_interval_{0};          // State of instance
  int64_t cur_data_retry_count_{0};  // State of data

  modelbox::Timer timer_;
  std::atomic_bool exit_flag_{false};
  std::mutex stop_lock_;
  std::condition_variable stop_cv_;

  int64_t retry_count_limit_{0};  // < 0 means unlimited, >= 0 means limited
  size_t retry_interval_base_ms_{0};
  size_t retry_interval_increment_ms_{0};
  size_t retry_interval_limit_ms_{0};
};

using BrokerInstances = std::map<std::string, std::shared_ptr<BrokerInstance>>;

class OutputBrokerFlowUnit : public modelbox::FlowUnit {
 public:
  OutputBrokerFlowUnit();
  ~OutputBrokerFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override;

  /* run when processing data */
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status DataPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status DataPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status DataGroupPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override {
    return modelbox::STATUS_OK;
  };

  modelbox::Status DataGroupPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override {
    return modelbox::STATUS_OK;
  };

 private:
  modelbox::Status SendData(std::shared_ptr<modelbox::DataContext> &data_ctx,
                            const std::string &output_broker_names,
                            const std::shared_ptr<modelbox::Buffer> &buffer);

  modelbox::Status ParseCfg(std::shared_ptr<modelbox::DataContext> &data_ctx,
                            const std::string &cfg);

  modelbox::Status InitBrokers(std::shared_ptr<modelbox::DataContext> &data_ctx,
                               const nlohmann::json &brokers_json);

  void AddBroker(std::shared_ptr<BrokerInstances> &broker_instances,
                 std::shared_ptr<BrokerNames> &broker_names,
                 const nlohmann::json &broker_json);

  std::shared_ptr<OutputBrokerPlugin> GetPlugin(const std::string &type);

  std::vector<std::shared_ptr<modelbox::DriverFactory>> factories_;
  std::map<std::string, std::shared_ptr<OutputBrokerPlugin>> plugins_;

  std::string mode_;
  int64_t retry_count_limit_{0};  // < 0 means unlimited, >= 0 means limited
  size_t retry_interval_base_ms_{0};
  size_t retry_interval_increment_ms_{0};
  size_t retry_interval_limit_ms_{0};
  size_t async_queue_size_{0};
};
#endif  // MODELBOX_FLOWUNIT_OUTPUT_BROKER_CPU_H_
