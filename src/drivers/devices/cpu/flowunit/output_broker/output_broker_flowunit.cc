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

#include "output_broker_flowunit.h"

#include <modelbox/base/timer.h>
#include <modelbox/base/utils.h>
#include <securec.h>

#include <queue>

#include "driver_util.h"
#include "modelbox/base/config.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

#define DEFAULT_RETRY_COUNT 5

BrokerDataQueue::BrokerDataQueue(const std::string &broker_name,
                                 size_t queue_size)
    : broker_name_(broker_name), queue_size_(queue_size) {}

void BrokerDataQueue::PushForce(
    const std::shared_ptr<modelbox::Buffer> &buffer) {
  std::lock_guard<std::mutex> lock(queue_lock_);
  while (queue_.size() >= queue_size_ && !queue_.empty()) {
    MBLOG_WARN << "Data in broker " << broker_name_ << " exceed limit "
               << queue_size_
               << ", old data drop one. set mode=\"sync\" will not drop data "
                  "but will stuck, "
                  "or you can enlarge queue_size in mode=\"async\".";
    queue_.pop();
  }

  queue_.push(buffer);
}

modelbox::Status BrokerDataQueue::Front(
    std::shared_ptr<modelbox::Buffer> &buffer) {
  std::lock_guard<std::mutex> lock(queue_lock_);
  if (queue_.empty()) {
    return modelbox::STATUS_NODATA;
  }

  buffer = queue_.front();
  return modelbox::STATUS_OK;
}

bool BrokerDataQueue::Empty() { return queue_.empty(); }

void BrokerDataQueue::PopIfEqual(
    const std::shared_ptr<modelbox::Buffer> &target) {
  std::lock_guard<std::mutex> lock(queue_lock_);
  if (queue_.empty()) {
    return;
  }

  if (queue_.front() != target) {
    return;
  }

  queue_.pop();
}

BrokerInstance::BrokerInstance(std::shared_ptr<OutputBrokerPlugin> &plugin,
                               const std::string &name,
                               std::shared_ptr<OutputBrokerHandle> &handle,
                               size_t async_queue_size)
    : plugin_(plugin),
      name_(name),
      handle_(handle),
      data_queue_(name, async_queue_size) {}

BrokerInstance::~BrokerInstance() {}

void BrokerInstance::SetRetryParam(int64_t retry_count_limit,
                                   size_t retry_interval_base_ms,
                                   size_t retry_interval_increment_ms,
                                   size_t retry_interval_limit_ms) {
  retry_count_limit_ = retry_count_limit;
  retry_interval_base_ms_ = retry_interval_base_ms;
  retry_interval_increment_ms_ = retry_interval_increment_ms;
  retry_interval_limit_ms_ = retry_interval_limit_ms;
}

modelbox::Status BrokerInstance::Write(
    const std::shared_ptr<modelbox::Buffer> &buffer) {
  cur_data_retry_count_ = 0;
  bool retry = true;
  do {
    auto ret = plugin_->Write(handle_, buffer);
    UpdateInstaceState(ret);
    if (ret == modelbox::STATUS_AGAIN) {
      if (cur_data_retry_count_++ < retry_count_limit_ ||
          retry_count_limit_ < 0) {
        MBLOG_ERROR << "Write data to " << name_
                    << " failed, detail: Try again";
      } else {
        retry = false;
      }

      if (send_interval_ != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(send_interval_));
      }
    } else {
      return ret;
    }
  } while (retry);
  return {modelbox::STATUS_FAULT,
          "Reach max retry limit " + std::to_string(retry_count_limit_)};
}

modelbox::Status BrokerInstance::AddToQueue(
    const std::shared_ptr<modelbox::Buffer> &buffer) {
  data_queue_.PushForce(buffer);
  std::lock_guard<std::mutex> lock(stop_lock_);
  if (is_stopped && !exit_flag_) {
    auto timer_task =
        std::make_shared<modelbox::TimerTask>([&]() { WriteFromQueue(); });
    is_stopped = false;
    timer_.Start();
    timer_.Schedule(timer_task, send_interval_, 0, true);
  }

  return modelbox::STATUS_OK;
}

void BrokerInstance::WriteFromQueue() {
  std::shared_ptr<modelbox::Buffer> buffer;
  data_queue_.Front(buffer);  // Will not be empty
  auto ret = plugin_->Write(handle_, buffer);
  UpdateInstaceState(ret);
  if (ret == modelbox::STATUS_AGAIN) {
    if (cur_data_retry_count_ < retry_count_limit_ || retry_count_limit_ < 0) {
      ++cur_data_retry_count_;
      MBLOG_ERROR << "Write data to " << name_ << " failed, detail: Try again ";
    } else {
      MBLOG_ERROR << "Write data to " << name_
                  << " failed, drop this data, detail: Reach max retry limit "
                  << retry_count_limit_;
      cur_data_retry_count_ = 0;
      data_queue_.PopIfEqual(buffer);
    }
  } else {
    if (!ret) {
      MBLOG_ERROR << "Write data to " << name_
                  << " failed, drop this data, detail: " << ret.Errormsg();
    } else {
      MBLOG_INFO << "Write data to " << name_ << " success";
    }

    cur_data_retry_count_ = 0;
    data_queue_.PopIfEqual(buffer);
  }

  std::lock_guard<std::mutex> lock(stop_lock_);
  if (!data_queue_.Empty()) {
    // if task stop, retry param will be changed by Dispose()
    auto timer_task =
        std::make_shared<modelbox::TimerTask>([&]() { WriteFromQueue(); });
    is_stopped = false;
    timer_.Start();
    timer_.Schedule(timer_task, send_interval_, 0, true);
  } else {
    is_stopped = true;
    stop_cv_.notify_all();
  }
}

void BrokerInstance::Dispose() {
  MBLOG_INFO << name_ << " start dispose";
  // set retry param to ensure task could exit
  if (retry_count_limit_ == -1) {
    retry_count_limit_ = DEFAULT_RETRY_COUNT;
  }
  retry_interval_increment_ms_ = 0;
  retry_interval_limit_ms_ = retry_interval_base_ms_;
  send_interval_ = retry_interval_base_ms_;
  // wait for sending task end
  exit_flag_ = true;
  std::unique_lock<std::mutex> lock(stop_lock_);
  stop_cv_.wait(lock, [&]() { return is_stopped.load(); });
  plugin_->Sync(handle_);
  plugin_->Close(handle_);
  MBLOG_INFO << name_ << " dispose over";
}

void BrokerInstance::UpdateInstaceState(modelbox::Status write_result) {
  switch (write_result.Code()) {
    case modelbox::STATUS_AGAIN:
      if (send_interval_ == 0) {
        send_interval_ = retry_interval_base_ms_;
      } else if (send_interval_ < retry_interval_limit_ms_) {
        send_interval_ += retry_interval_increment_ms_;
        if (send_interval_ > retry_interval_limit_ms_) {
          send_interval_ = retry_interval_limit_ms_;
        }
      }

      break;

    default:
      send_interval_ = 0;
      break;
  }
}

OutputBrokerFlowUnit::OutputBrokerFlowUnit(){};
OutputBrokerFlowUnit::~OutputBrokerFlowUnit(){};

modelbox::Status OutputBrokerFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  auto dev_mgr = GetBindDevice()->GetDeviceManager();
  if (dev_mgr == nullptr) {
    MBLOG_ERROR << "Can not get device manger";
    return modelbox::STATUS_FAULT;
  }

  auto drivers = dev_mgr->GetDrivers();
  if (drivers == nullptr) {
    MBLOG_ERROR << "Can not get drivers";
    return modelbox::STATUS_FAULT;
  }

  auto ret = driverutil::GetPlugin<OutputBrokerPlugin>(
      DRIVER_CLASS_OUTPUT_BROKER_PLUGIN, drivers, factories_, plugins_);
  if (!ret) {
    return ret;
  }

  for (auto &item : plugins_) {
    auto ret = item.second->Init(opts);
    if (!ret) {
      MBLOG_ERROR << "Init plugin " << item.first
                  << " failed, detail : " << ret.Errormsg();
    }
  }

  mode_ = opts->GetString("mode", SYNC_MODE);
  if (mode_ != SYNC_MODE && mode_ != ASYNC_MODE) {
    MBLOG_ERROR << "Mode only support {sync, async}";
    return modelbox::STATUS_BADCONF;
  }

  retry_count_limit_ = opts->GetInt64("retry_count_limit");
  retry_interval_base_ms_ = opts->GetUint64("retry_interval_base_ms");
  retry_interval_increment_ms_ = opts->GetUint64("retry_interval_increment_ms");
  retry_interval_limit_ms_ =
      opts->GetUint64("retry_interval_limit_ms", retry_interval_base_ms_);
  if (retry_interval_limit_ms_ < retry_interval_base_ms_) {
    MBLOG_WARN << "retry_interval_limit < retry_interval_base is unacceptable, "
                  "use retry_interval_base as retry_interval_limit";
    retry_interval_limit_ms_ = retry_interval_base_ms_;
  }

  async_queue_size_ = opts->GetUint64("queue_size", 100);
  return modelbox::STATUS_OK;
}

modelbox::Status OutputBrokerFlowUnit::Close() {
  for (auto &item : plugins_) {
    item.second->Deinit();
  }

  plugins_.clear();

  return modelbox::STATUS_OK;
}

modelbox::Status OutputBrokerFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto input_buffer_list = data_ctx->Input(INPUT_DATA);
  for (auto &buffer : *input_buffer_list) {
    std::string output_broker_names;
    buffer->Get(META_OUTPUT_BROKER_NAME, output_broker_names);
    auto ret = SendData(data_ctx, output_broker_names, buffer);
    if (!ret) {
      MBLOG_ERROR << "Send data to output broker " << output_broker_names
                  << "failed";
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status OutputBrokerFlowUnit::SendData(
    std::shared_ptr<modelbox::DataContext> &data_ctx,
    const std::string &output_broker_names,
    const std::shared_ptr<modelbox::Buffer> &buffer) {
  auto broker_instances = std::static_pointer_cast<BrokerInstances>(
      data_ctx->GetPrivate(CTX_BROKER_INSTANCES));
  auto loaded_broker_names = std::static_pointer_cast<BrokerNames>(
      data_ctx->GetPrivate(CTX_BROKER_NAMES));
  if (broker_instances == nullptr || loaded_broker_names == nullptr) {
    MBLOG_ERROR << "Output broker handles has not been inited";
    return modelbox::STATUS_FAULT;
  }

  auto output_broker_name_list =
      modelbox::StringSplit(output_broker_names, '|');
  if (output_broker_name_list.empty()) {
    output_broker_name_list = *loaded_broker_names;
  }

  for (auto &target_broker_name : output_broker_name_list) {
    auto item = broker_instances->find(target_broker_name);
    if (item == broker_instances->end()) {
      MBLOG_ERROR << "Wrong broker name " << target_broker_name
                  << ", it's not named in config";
      continue;
    }

    auto &broker = item->second;
    if (mode_ == SYNC_MODE) {
      auto ret = broker->Write(buffer);
      if (!ret) {
        MBLOG_ERROR << "Write data to " << target_broker_name
                    << " failed, drop this data, detail: " << ret.Errormsg();
      } else {
        MBLOG_INFO << "Write data to " << target_broker_name << " success";
      }
    } else {
      broker->AddToQueue(buffer);
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status OutputBrokerFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto config = data_ctx->GetSessionConfig();
  auto cfg_str = config->GetString(SESSION_OUTPUT_BROKER_CONFIG);
  if (cfg_str.empty()) {
    MBLOG_ERROR << "Output broker config in session has not been set";
    return modelbox::STATUS_FAULT;
  }

  auto ret = ParseCfg(data_ctx, cfg_str);
  if (!ret) {
    MBLOG_ERROR << "Parse output broker config failed";
    return ret;
  }

  return modelbox::STATUS_OK;
};

modelbox::Status OutputBrokerFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto broker_instances = std::static_pointer_cast<BrokerInstances>(
      data_ctx->GetPrivate(CTX_BROKER_INSTANCES));
  for (auto &item : *broker_instances) {
    item.second->Dispose();
  }

  broker_instances->clear();
  return modelbox::STATUS_OK;
};

modelbox::Status OutputBrokerFlowUnit::ParseCfg(
    std::shared_ptr<modelbox::DataContext> &data_ctx, const std::string &cfg) {
  nlohmann::json json;
  try {
    json = nlohmann::json::parse(cfg);
  } catch (const std::exception &e) {
    MBLOG_ERROR << "Parse output config to json failed, detail: " << e.what();
    return modelbox::STATUS_INVALID;
  }

  if (!json.is_object()) {
    MBLOG_ERROR << "Output broker config must a json object";
    return modelbox::STATUS_INVALID;
  }

  auto brokers = json["brokers"];
  auto ret = InitBrokers(data_ctx, brokers);
  if (!ret) {
    MBLOG_ERROR << "Init output brokers failed";
    return ret;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status OutputBrokerFlowUnit::InitBrokers(
    std::shared_ptr<modelbox::DataContext> &data_ctx,
    const nlohmann::json &brokers_json) {
  if (brokers_json.empty()) {
    MBLOG_ERROR << "Key <brokers> is missing in json object";
    return modelbox::STATUS_INVALID;
  }

  if (!brokers_json.is_array()) {
    MBLOG_ERROR << "Value of <brokers> must be an array";
    return modelbox::STATUS_INVALID;
  }

  auto broker_instances = std::make_shared<BrokerInstances>();
  auto broker_names = std::make_shared<BrokerNames>();
  for (auto &broker_json : brokers_json) {
    try {
      AddBroker(broker_instances, broker_names, broker_json);
    } catch (const std::exception &e) {
      MBLOG_ERROR << "init output broker failed, config: "
                  << broker_json.dump();
      return modelbox::STATUS_INVALID;
    }
  }

  data_ctx->SetPrivate(CTX_BROKER_INSTANCES, broker_instances);
  data_ctx->SetPrivate(CTX_BROKER_NAMES, broker_names);
  return modelbox::STATUS_OK;
}

void OutputBrokerFlowUnit::AddBroker(
    std::shared_ptr<BrokerInstances> &broker_instances,
    std::shared_ptr<BrokerNames> &broker_names,
    const nlohmann::json &broker_json) {
  if (!broker_json.is_object()) {
    MBLOG_ERROR << "Single broker config must be object";
    return;
  }

  auto type = broker_json["type"];
  if (type.empty()) {
    MBLOG_WARN << "Key <type> is missing in single broker config";
    return;
  }

  if (!type.is_string()) {
    MBLOG_WARN << "Key <type> must has string value";
    return;
  }

  auto plugin = GetPlugin(type);
  if (plugin == nullptr) {
    MBLOG_WARN << "No ouput broker plugin for type " << type;
    return;
  }

  auto name = broker_json["name"];
  if (name.empty()) {
    MBLOG_WARN << "Key <name> is missing in single broker config, type "
               << type;
    return;
  }

  if (!name.is_string()) {
    MBLOG_WARN << "Key <name> must has string value, type " << type;
    return;
  }

  auto cfg = broker_json["cfg"];
  if (cfg.empty()) {
    MBLOG_WARN << "Key <cfg> is missing in single broker config, type " << type
               << ", name " << name;
    return;
  }

  if (!cfg.is_string()) {
    MBLOG_WARN << "Key <cfg> must has string value, type " << type << ", name "
               << name;
    return;
  }

  auto handle = plugin->Open(cfg);
  if (handle == nullptr) {
    MBLOG_WARN << "Get broker handle for " << name << ":" << type << " failed";
    return;
  }

  handle->output_broker_type_ = type;
  auto instance =
      std::make_shared<BrokerInstance>(plugin, name, handle, async_queue_size_);
  instance->SetRetryParam(retry_count_limit_, retry_interval_base_ms_,
                          retry_interval_increment_ms_,
                          retry_interval_limit_ms_);
  (*broker_instances)[name] = instance;
  broker_names->push_back(name);
}

std::shared_ptr<OutputBrokerPlugin> OutputBrokerFlowUnit::GetPlugin(
    const std::string &type) {
  auto item = plugins_.find(type);
  if (item == plugins_.end()) {
    return nullptr;
  }

  return item->second;
}

MODELBOX_FLOWUNIT(OutputBrokerFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Output");
  desc.AddFlowUnitInput({INPUT_DATA});
  desc.SetFlowType(modelbox::FlowType::STREAM);
  desc.SetInputContiguous(false);
  desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
