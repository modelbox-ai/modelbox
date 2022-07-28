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


#include "modelbox/profiler.h"

#include <fstream>
#include <nlohmann/json.hpp>

namespace modelbox {

class PerfCollector {
 public:
  virtual void Export(nlohmann::json& perf_data) = 0;

  virtual void Collect() = 0;

  virtual bool Empty() = 0;
};

class CpuUsageData {
 public:
  CpuUsageData(TimePoint timestamp, int32_t percentage)
      : timestamp_(timestamp), percentage_(percentage) {}

  virtual ~CpuUsageData() {}
  TimePoint timestamp_;
  int32_t percentage_{0};  // 0 ~ 100
};


class DeviceMemUsageData {
 public:
  DeviceMemUsageData(const std::string& device_tag, int64_t device_mem,
                     int32_t device_mem_percentage)
      : device_tag_(device_tag),
        device_mem_(device_mem),
        device_mem_percentage_(device_mem_percentage) {}
  virtual ~DeviceMemUsageData() {}

  std::string device_tag_;
  int64_t device_mem_{0};
  int32_t device_mem_percentage_{0};
};

class MemUsageData {
 public:
  MemUsageData(TimePoint timestamp, int64_t host_mem,
               int32_t host_mem_percentage)
      : timestamp_(timestamp),
        host_mem_(host_mem),
        host_mem_percentage_(host_mem_percentage) {}
  virtual ~MemUsageData() {}

  void AddDeviceMemUsageData(const std::string& device_tag, int64_t device_mem,
                             int32_t device_mem_percentage) {
    device_mem_data_list_.emplace_back(device_tag, device_mem,
                                       device_mem_percentage);
  }

  TimePoint timestamp_;
  int64_t host_mem_{0};
  int32_t host_mem_percentage_{0};
  std::list<DeviceMemUsageData> device_mem_data_list_;
};

class MemUsageCollector : public PerfCollector {
 public:
  MemUsageCollector(std::shared_ptr<
                    std::map<std::string, std::pair<std::string, std::string>>>
                        devices)
      : devices_(devices){};

  virtual ~MemUsageCollector() {}

  void Export(nlohmann::json& perf_data) override;

  void Collect() override;

  bool Empty() override;

 private:
  // device type + device id -> std::map<TimePoint, int32_t>
  std::list<MemUsageData> mem_usage_data_list_;
  std::mutex data_mutex_;
  std::shared_ptr<std::map<std::string, std::pair<std::string, std::string>>>
      devices_;
};

class FlowUnitPerfCollector : public PerfCollector {
 public:
  FlowUnitPerfCollector(
      std::shared_ptr<
          std::map<std::string, std::pair<std::string, std::string>>>
          devices,
      std::shared_ptr<std::vector<std::string>> flow_unit_names);
  virtual ~FlowUnitPerfCollector();

  void Export(nlohmann::json& perf_data) override;

  void Collect() override;

  bool Empty() override;

  std::shared_ptr<FlowUnitPerfCtx> GetFlowUnitPerfCtx(
      const std::string& flow_unit_name);

 private:
  std::mutex data_mutex_;
  // FlowUnit name -> FlowUnitProfile
  std::map<std::string, std::shared_ptr<FlowUnitPerfCtx>>
      flow_unit_per_ctx_map_;
  std::shared_ptr<std::map<std::string, std::pair<std::string, std::string>>>
      devices_;
  std::shared_ptr<std::vector<std::string>> flow_unit_names_;
};

class CpuUsageCollector : public PerfCollector {
 public:
  CpuUsageCollector() = default;

  virtual ~CpuUsageCollector() = default;

  void Export(nlohmann::json& perf_data) override;

  void Collect() override;

  bool Empty() override;

 private:
  std::list<CpuUsageData> cpu_usage_data_list;
  std::mutex data_mutex_;
};

Performance::Performance(std::shared_ptr<DeviceManager> device_mgr,
                         std::string& output_dir_path)
    : ProfilerLifeCycle("Performance"),
      timer_(nullptr),
      sample_interval_(DEFAULT_TIMER_SAMPLE_INTERVAL),
      write_file_interval_(DEFAULT_WRITE_PROFILE_INTERVAL),
      device_mgr_(device_mgr),
      output_dir_path_(output_dir_path) {}
Performance::~Performance() {
  if (IsRunning()) {
    Stop();
  }
}

Status Performance::OnInit() {
  if (device_mgr_ == nullptr) {
    return STATUS_FAULT;
  }

  devices_ = std::make_shared<
      std::map<std::string, std::pair<std::string, std::string>>>();

  flow_unit_names_ = std::make_shared<std::vector<std::string>>();

  auto device_map = device_mgr_->GetDeviceList();
  for (auto devices : device_map) {
    for (auto device : devices.second) {
      std::string device_type = devices.first;
      std::string device_id = device.first;
      devices_->insert(std::make_pair(device_type + device_id,
                                      std::make_pair(device_type, device_id)));
    }
  }

  // TODO : get flow unit names from session context

  auto cpu_usage_collector = std::make_shared<CpuUsageCollector>();
  perf_collectors_.push_back(cpu_usage_collector);

  auto mem_usage_collector = std::make_shared<MemUsageCollector>(devices_);
  perf_collectors_.push_back(mem_usage_collector);

  auto flow_unit_perf_collector =
      std::make_shared<FlowUnitPerfCollector>(devices_, flow_unit_names_);
  perf_collectors_.push_back(flow_unit_perf_collector);

  flow_unit_perf_collector_ = flow_unit_perf_collector;
  return STATUS_SUCCESS;
}

Status Performance::OnStart() {
  timer_run_ = true;
  timer_ = std::make_shared<std::thread>(&Performance::PerformanceWorker, this);
  return STATUS_SUCCESS;
}

Status Performance::OnStop() {
  OnPause();

  WritePerformance();
  return STATUS_SUCCESS;
}

Status Performance::OnPause() {
  if (timer_) {
    timer_run_ = false;
    timer_->join();
    timer_ = nullptr;
  }

  return STATUS_SUCCESS;
}

Status Performance::OnResume() { return OnStart(); }

Status Performance::WritePerformance() {
  bool is_empty = true;
  for (const auto& collector : perf_collectors_) {
    if (!collector->Empty()) {
      is_empty = false;
      break;
    }
  }

  if (is_empty) {
    return STATUS_SUCCESS;
  }

  nlohmann::json perf_json;
  for (const auto& collector : perf_collectors_) {
    collector->Export(perf_json);
  }

  time_t current_time = time(nullptr);
  char buf[64] = {0};
  auto* local_tm = localtime(&current_time);
  if (local_tm) {
    strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", local_tm);
  }

  // TODO: graph_name + task_name + timestample
  std::string file_path =
      output_dir_path_ + "/" + "performance_" + std::string(buf) + ".json";

  std::ofstream out(file_path);
  if (out.is_open() == false) {
    MBLOG_ERROR << "write trace failed, file path : " << file_path;
    return STATUS_FAULT;
  }

  std::string profiles_json_str = perf_json.dump();
  out.write(profiles_json_str.c_str(), profiles_json_str.size());
  if (out.rdstate() & std::ios::failbit) {
    MBLOG_ERROR << "Write file " << file_path << " failed";
    out.close();
    return STATUS_FAULT;
  }

  out.close();

  return STATUS_SUCCESS;
}

std::shared_ptr<FlowUnitPerfCtx> Performance::GetFlowUnitPerfCtx(
    const std::string& flow_unit_name) {
  return flow_unit_perf_collector_->GetFlowUnitPerfCtx(flow_unit_name);
}

void Performance::PerformanceWorker() {
  unsigned long now = {0};
  int32_t sleep = sample_interval_;
  int32_t sleep_time = 0;
  unsigned long expect_time = 0;

  MBLOG_INFO << "profiler timer start";

  sleep_time = sleep;
  now = GetTickCount();
  expect_time = now + sleep;

  uint32_t count = 0;
  while (timer_run_) {
    now = GetTickCount();
    sleep_time = expect_time - now;
    if (sleep_time < 0) {
      sleep_time = 0;
      expect_time = now;
    }

    expect_time += sleep;
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));

    for (auto profile_data : perf_collectors_) {
      profile_data->Collect();
    }

    count++;
    if (count > write_file_interval_) {
      WritePerformance();
      count = 0;
    }
  }

  MBLOG_INFO << "profiler timer end";
}

void Performance::SetTimerSampleInterval(int32_t interval) {
  sample_interval_ = interval;
}

void Performance::SetWriteFileInterval(int32_t interval) {
  write_file_interval_ = interval;
}

FlowUnitPerfCtx::FlowUnitPerfCtx(const std::string& flow_unit_name) {
  flow_unit_name_ = flow_unit_name;
  process_latency_ = 0;
  process_latency_count_ = 0;
};

void FlowUnitPerfCtx::UpdateProcessLatency(int32_t process_latency) {
  std::lock_guard<std::mutex> lock(latency_mutex_);
  double total_latency = process_latency_ * process_latency_count_;
  total_latency += process_latency;
  process_latency_count_++;
  process_latency_ = total_latency / process_latency_count_;
}

int32_t FlowUnitPerfCtx::GetProcessLatency() {
  return static_cast<int32_t>(process_latency_);
}

void FlowUnitPerfCtx::UpdateDeviceMemory(std::string& device_type,
                                         std::string& device_id,
                                         int32_t memory) {
  std::lock_guard<std::mutex> lock(devices_memories_mutex_);
  std::string device = device_type + device_id;
  if (devices_memories_.find(device) == devices_memories_.end()) {
    std::map<TimePoint, int32_t> device_memories;
    devices_memories_.insert(std::make_pair(device, device_memories));
  }

  auto current_time = std::chrono::time_point_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now());
  devices_memories_[device].insert(std::make_pair(current_time, memory));
}

int32_t FlowUnitPerfCtx::GetDeviceMemory(std::string& device_type,
                                         std::string& device_id) {
  std::lock_guard<std::mutex> lock(devices_memories_mutex_);
  std::string device = device_type + device_id;
  if (devices_memories_.find(device) == devices_memories_.end()) {
    MBLOG_ERROR
        << "can not get device memory profiling information, device_type : "
        << device_type << ", device_id : " << device_id;
    return 0;
  }

  auto device_memories = devices_memories_[device];
  if (device_memories.empty()) {
    return 0;
  }

  return device_memories.rbegin()->second;
}

void FlowUnitPerfCtx::UpdateDeviceMemoryUsage(std::string& device_type,
                                              std::string& device_id,
                                              int32_t memory_usage) {
  std::lock_guard<std::mutex> lock(devices_memories_usage_mutex_);
  std::string device = device_type + device_id;
  if (devices_memories_usage_.find(device) == devices_memories_usage_.end()) {
    std::map<TimePoint, int32_t> device_memories_usage;
    devices_memories_usage_.insert(
        std::make_pair(device, device_memories_usage));
  }

  auto current_time = std::chrono::time_point_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now());
  devices_memories_usage_[device].insert(
      std::make_pair(current_time, memory_usage));
}

int32_t FlowUnitPerfCtx::GetDeviceMemoryUsage(std::string& device_type,
                                              std::string& device_id) {
  std::lock_guard<std::mutex> lock(devices_memories_usage_mutex_);
  std::string device = device_type + device_id;
  if (devices_memories_usage_.find(device) == devices_memories_usage_.end()) {
    MBLOG_ERROR << "can not get device memory usage profiling information, "
                   "device_type : "
                << device_type << ", device_id : " << device_id;
    return 0;
  }

  auto device_memories_usage = devices_memories_usage_[device];
  if (device_memories_usage.empty()) {
    return 0;
  }

  return device_memories_usage.rbegin()->second;
}

void CpuUsageCollector::Export(nlohmann::json& perf_data) {
  data_mutex_.lock();
  nlohmann::json cpu_usage_json_arr = nlohmann::json::array();
  for (const auto& data : cpu_usage_data_list) {
    nlohmann::json cpu_usage_json;
    cpu_usage_json["timestamp"] = data.timestamp_.time_since_epoch().count();
    cpu_usage_json["percentage"] = data.percentage_;
    cpu_usage_json_arr.push_back(cpu_usage_json);
  }

  data_mutex_.unlock();
  perf_data["cpu_usage"] = cpu_usage_json_arr;

  cpu_usage_data_list.clear();
}

void CpuUsageCollector::Collect() {
  auto current_time = std::chrono::time_point_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now());
  int32_t cpu_percentage = 0;
  data_mutex_.lock();
  cpu_usage_data_list.emplace_back(current_time, cpu_percentage);
  data_mutex_.unlock();
}

bool CpuUsageCollector::Empty() { return cpu_usage_data_list.empty(); }

void MemUsageCollector::Export(nlohmann::json& perf_data) {
  data_mutex_.lock();
  nlohmann::json mem_usage_json_arr = nlohmann::json::array();
  for (const auto& data : mem_usage_data_list_) {
    nlohmann::json mem_usage_json;
    mem_usage_json["timestamp"] = data.timestamp_.time_since_epoch().count();
    mem_usage_json["host_mem"] = data.host_mem_;
    mem_usage_json["host_mem_percentage"] = data.host_mem_percentage_;
    nlohmann::json device_mem_usage_json_arr = nlohmann::json::array();
    for (const auto& dev_mem_data : data.device_mem_data_list_) {
      nlohmann::json device_mem_usage_json;
      device_mem_usage_json["device"] = dev_mem_data.device_tag_;
      device_mem_usage_json["device_mem"] = dev_mem_data.device_mem_;
      device_mem_usage_json["device_mem_percentage"] =
          dev_mem_data.device_mem_percentage_;
      device_mem_usage_json_arr.push_back(device_mem_usage_json);
    }

    mem_usage_json["device_memory"] = device_mem_usage_json_arr;
    mem_usage_json_arr.push_back(mem_usage_json);
  }

  data_mutex_.unlock();
  perf_data["memory_usage"] = mem_usage_json_arr;
  mem_usage_data_list_.clear();
}

void MemUsageCollector::Collect() {
  auto current_time = std::chrono::time_point_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now());
  MemUsageData data(current_time, 0, 0);
  for (auto device : *devices_) {
    std::string device_type = device.second.first;
    std::string device_id = device.second.second;
    std::string device_tag = device_type + ":" + device_id;
    int32_t memory = 0;
    int32_t memory_percentage = 0;
    data.AddDeviceMemUsageData(device_tag, memory, memory_percentage);
  }

  data_mutex_.lock();
  mem_usage_data_list_.push_back(data);
  data_mutex_.unlock();
}

bool MemUsageCollector::Empty() { return mem_usage_data_list_.empty(); }

FlowUnitPerfCollector::FlowUnitPerfCollector(
    std::shared_ptr<std::map<std::string, std::pair<std::string, std::string>>>
        devices,
    std::shared_ptr<std::vector<std::string>> flow_unit_names)
    : devices_(devices), flow_unit_names_(flow_unit_names) {}

void FlowUnitPerfCollector::Export(nlohmann::json& perf_data) {
  data_mutex_.lock();
  nlohmann::json flow_unit_perf_json_arr = nlohmann::json::array();
  for (auto item : flow_unit_per_ctx_map_) {
    nlohmann::json flow_unit_perf_json;
    flow_unit_perf_json["flow_unit_name"] = item.first;
    flow_unit_perf_json["process_latency"] = item.second->GetProcessLatency();

    flow_unit_perf_json_arr.push_back(flow_unit_perf_json);
  }

  data_mutex_.unlock();
  perf_data["flow_unit_performance"] = flow_unit_perf_json_arr;
}

FlowUnitPerfCollector::~FlowUnitPerfCollector() {}

void FlowUnitPerfCollector::Collect() {
  for (auto device : *devices_) {
    std::string device_type = device.second.first;
    std::string device_id = device.second.second;
    for (auto flow_unit_name : *flow_unit_names_) {
    }
  }
}

bool FlowUnitPerfCollector::Empty() { return flow_unit_per_ctx_map_.empty(); }

std::shared_ptr<FlowUnitPerfCtx> FlowUnitPerfCollector::GetFlowUnitPerfCtx(
    const std::string& flow_unit_name) {
  std::lock_guard<std::mutex> perf_lock(data_mutex_);
  if (flow_unit_per_ctx_map_.find(flow_unit_name) ==
      flow_unit_per_ctx_map_.end()) {
    auto flow_unit_perf_ctx = std::make_shared<FlowUnitPerfCtx>(flow_unit_name);
    flow_unit_per_ctx_map_[flow_unit_name] = flow_unit_perf_ctx;
    return flow_unit_perf_ctx;
  }

  return flow_unit_per_ctx_map_[flow_unit_name];
}
}  // namespace modelbox