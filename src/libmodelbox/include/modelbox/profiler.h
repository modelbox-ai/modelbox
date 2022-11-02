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

#ifndef MODELBOX_PROFLER_H_
#define MODELBOX_PROFLER_H_

#include <modelbox/base/any.h>
#include <modelbox/base/configuration.h>
#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/base/thread_pool.h>
#include <modelbox/base/timer.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace modelbox {

constexpr const char* PROFILE_PATH_ENV = "PROFILE_PATH";

enum class TraceSliceType {
  OPEN,
  CLOSE,
  PROCESS,
  STREAM_OPEN,
  STREAM_CLOSE,
  CUSTOM
};

enum class EventType { BEGIN, END };

using TimePoint = std::chrono::time_point<std::chrono::system_clock,
                                          std::chrono::microseconds>;

constexpr uint32_t DEFAULT_TIMER_SAMPLE_INTERVAL = 100;

constexpr uint32_t DEFAULT_WRITE_PROFILE_INTERVAL = 600;

constexpr uint32_t DEFAULT_WRITE_TRACE_INTERVAL = 600;

class ProfilerLifeCycle {
 public:
  ProfilerLifeCycle(std::string name);

  virtual ~ProfilerLifeCycle();

  Status Init();

  Status Start();

  Status Stop();

  Status Pause();

  Status Resume();

  inline bool IsRunning() { return is_running_; }

  inline bool IsInitialized() { return is_initialized_; }

 protected:
  virtual Status OnInit() { return STATUS_SUCCESS; };

  virtual Status OnStart() { return STATUS_SUCCESS; };

  virtual Status OnStop() { return STATUS_SUCCESS; };

  virtual Status OnPause() { return STATUS_SUCCESS; };

  virtual Status OnResume() { return STATUS_SUCCESS; };

 private:
  std::string name_;
  std::atomic_bool is_running_{false};
  std::atomic_bool is_initialized_{false};
};

class TraceEvent {
  friend class TraceSlice;

 public:
  virtual ~TraceEvent();

  TraceEvent& SetEventType(const EventType& event_type);

  const EventType& GetEventType() const;

  TraceEvent& SetEventTime(const TimePoint& event_time);

  const TimePoint& GetEventTime() const;

  TraceEvent& SetThreadId(std::thread::id thread_id);

  std::thread::id GetThreadId() const;

 protected:
  TraceEvent();

 private:
  EventType event_type_{EventType::BEGIN};
  TimePoint event_time_;
  std::thread::id thread_id_;
};

class FlowUnitTrace;
class FlowUnitPerfCtx;

class TraceSlice {
  friend class FlowUnitTrace;

 public:
  virtual ~TraceSlice();

  void Begin();

  void End();

  inline std::shared_ptr<TraceEvent> GetBeginEvent() {
    return begin_event_ptr_;
  }

  inline std::shared_ptr<TraceEvent> GetEndEvent() { return end_event_ptr_; };

  inline TraceSliceType GetTraceSliceType() { return slice_type_; }

  int32_t GetDuration();

  std::string GetSession();

  inline void SetBatchSize(uint32_t batch_size) { batch_size_ = batch_size; }

  inline uint32_t GetBatchSize() { return batch_size_; }

 protected:
  TraceSlice(TraceSliceType& slice_type, std::string session,
             const std::shared_ptr<FlowUnitTrace>& flow_unit_trace_ptr,
             std::shared_ptr<FlowUnitPerfCtx> flow_unit_perf_ctx);

  TraceSlice(TraceSliceType& slice_type, std::string session,
             const std::shared_ptr<FlowUnitTrace>& flow_unit_trace_ptr,
             std::shared_ptr<TraceEvent> begin,
             std::shared_ptr<TraceEvent> end);

 private:
  TraceSliceType slice_type_;
  std::string session_;
  std::weak_ptr<FlowUnitTrace> flow_unit_trace_ptr_;
  std::shared_ptr<FlowUnitPerfCtx> flow_unit_perf_ctx_;
  std::shared_ptr<TraceEvent> begin_event_ptr_;
  std::shared_ptr<TraceEvent> end_event_ptr_;
  bool is_end_called_;
  uint32_t batch_size_;
};

class Trace;

class FlowUnitTrace : public std::enable_shared_from_this<FlowUnitTrace> {
  friend class Trace;

 public:
  virtual ~FlowUnitTrace();

  inline const std::string& GetFlowUnitName() const { return flow_unit_name_; }

  // return a new TraceSlice when call this function, if Slice has not
  // couple begin and end, it will be ignored
  std::shared_ptr<TraceSlice> Slice(TraceSliceType slice_type,
                                    std::string session);

  void GetTraceSlices(std::vector<std::shared_ptr<TraceSlice>>& trace_slices);

  Status AddTraceSlice(const std::shared_ptr<TraceSlice>& trace_slice);

  void SetFlowUnitPerfCtx(std::shared_ptr<FlowUnitPerfCtx> flow_unit_perf_ctx);

 protected:
  explicit FlowUnitTrace(std::string flow_unit_name);

 private:
  std::string flow_unit_name_;
  std::shared_ptr<FlowUnitPerfCtx> flow_unit_perf_ctx_;
  std::vector<std::shared_ptr<TraceSlice>> trace_slices_;
  std::mutex trace_slices_mutex_;
};

class FlowUnitPerfCtx {
 public:
  explicit FlowUnitPerfCtx(const std::string& flow_unit_name);

  virtual ~FlowUnitPerfCtx();

  void UpdateProcessLatency(int32_t process_latency);

  int32_t GetProcessLatency();

  void UpdateDeviceMemory(std::string& device_type, std::string& device_id,
                          int32_t memory);

  int32_t GetDeviceMemory(std::string& device_type, std::string& device_id);

  void UpdateDeviceMemoryUsage(std::string& device_type, std::string& device_id,
                               int32_t memory_usage);

  int32_t GetDeviceMemoryUsage(std::string& device_type,
                               std::string& device_id);

  inline std::map<std::string, std::map<TimePoint, int32_t>>
  GetDeviceMemoryMap() {
    std::lock_guard<std::mutex> lock(devices_memories_mutex_);
    return devices_memories_;
  }

  inline std::map<std::string, std::map<TimePoint, int32_t>>
  GetDeviceMemoryUsageMap() {
    std::lock_guard<std::mutex> lock(devices_memories_usage_mutex_);
    return devices_memories_usage_;
  }

 private:
  std::string flow_unit_name_;
  double process_latency_;
  int32_t process_latency_count_;

  // device type + id -> std::map<TimePoint int32_t>
  std::map<std::string, std::map<TimePoint, int32_t>> devices_memories_;
  std::map<std::string, std::map<TimePoint, int32_t>> devices_memories_usage_;

  std::mutex latency_mutex_;
  std::mutex devices_memories_mutex_;
  std::mutex devices_memories_usage_mutex_;
};

class FlowUnitPerfCollector;
class PerfCollector;

class Performance : public ProfilerLifeCycle {
 public:
  Performance(std::shared_ptr<DeviceManager> device_mgr,
              std::string& output_dir_path);
  ~Performance() override;

  Status OnInit() override;

  Status OnStart() override;

  Status OnStop() override;

  Status OnPause() override;

  Status OnResume() override;

  void SetTimerSampleInterval(int32_t interval);

  void SetWriteFileInterval(int32_t interval);

  Status WritePerformance();

  std::shared_ptr<FlowUnitPerfCtx> GetFlowUnitPerfCtx(
      const std::string& flow_unit_name);

 private:
  // process statics, get by regular sampling
  int32_t GetProcessDeviceMemory(std::string& device_type,
                                 std::string& device_id);

  int32_t GetProcessDeviceMemoryUsage(std::string& device_type,
                                      std::string& device_id);

  int32_t GetProcessCpuUsage();

  // flow unit statics, get by regular sampling
  int32_t GetFlowUnitDeviceMemory(std::string& device_type,
                                  std::string& device_id,
                                  std::string& flow_unit_name);

  void PerformanceWorker();

  // device type + device id -> std::pair<std::string, std::string>
  std::shared_ptr<std::map<std::string, std::pair<std::string, std::string>>>
      devices_;

  std::shared_ptr<std::vector<std::string>> flow_unit_names_;

  std::atomic_bool timer_run_{false};

  std::shared_ptr<std::thread> timer_;

  uint32_t sample_interval_;

  uint32_t write_file_interval_;

  std::shared_ptr<DeviceManager> device_mgr_;

  std::string output_dir_path_;

  std::shared_ptr<FlowUnitPerfCollector> flow_unit_perf_collector_;

  std::vector<std::shared_ptr<PerfCollector>> perf_collectors_;
};

class Trace : public std::enable_shared_from_this<Trace>,
              public ProfilerLifeCycle {
 public:
  Trace(std::string output_dir_path, std::shared_ptr<Performance> perf,
        bool session_enable);
  ~Trace() override;

  Status OnStart() override;

  Status OnStop() override;

  Status OnPause() override;

  Status OnResume() override;

  std::shared_ptr<FlowUnitTrace> FlowUnit(const std::string& flow_unit_name);

  Status WriteTrace();

  void SetWriteFileInterval(int32_t threshold);

  uint32_t GetWriteFileInterval();

  void SetSessionEnable();

 private:
  std::string TraceSliceTypeToString(TraceSliceType type);

  void TraceWork();

  // FlowUnit name -> FlowUnitTrace, get by lock
  std::map<std::string, std::shared_ptr<FlowUnitTrace>> traces_;

  std::mutex trace_mutex_;

  std::string output_dir_path_;

  std::shared_ptr<Performance> perf_;

  uint32_t write_file_interval_;

  std::atomic_bool timer_run_{false};

  std::shared_ptr<std::thread> timer_;

  std::atomic_bool session_enable_;
};

/**
 * call as following in one session:

 * auto trace =
 * profiler->FlowUint("resize")->Slice(TraceSliceType::PROCESS);
 * trace->Begin();
 * process();
 * trace->End();
 */
class Profiler : public ProfilerLifeCycle {
 public:
  explicit Profiler(std::shared_ptr<DeviceManager> device_mgr,
                    std::shared_ptr<Configuration> config);

  ~Profiler() override;

  Profiler(const Profiler& profiler) = delete;
  Profiler& operator=(const Profiler& profiler) = delete;
  Profiler(const Profiler&& profiler) = delete;
  Profiler& operator=(const Profiler&& profiler) = delete;

  Status OnInit() override;

  Status InitProfilerDir();

  Status OnStart() override;

  Status OnStop() override;

  Status OnPause() override;

  Status OnResume() override;

  std::shared_ptr<Performance> GetPerf();

  std::shared_ptr<Trace> GetTrace();

 private:
  std::shared_ptr<DeviceManager> device_mgr_;

  std::shared_ptr<Configuration> config_;

  std::string output_dir_path_;

  std::shared_ptr<Performance> perf_;

  std::shared_ptr<Trace> trace_;
};

}  // namespace modelbox
#endif  // MODELBOX_PROFLER_H_
