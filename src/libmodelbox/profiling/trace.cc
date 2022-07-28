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


#include <fstream>
#include <nlohmann/json.hpp>

#include "modelbox/profiler.h"

namespace modelbox {
const std::map<TraceSliceType, std::string> TRACE_SLICE_TYPE = {
    {TraceSliceType::OPEN, "OPEN"},
    {TraceSliceType::CLOSE, "CLOSE"},
    {TraceSliceType::PROCESS, "PROCESS"},
    {TraceSliceType::STREAM_OPEN, "STREAM_OPEN"},
    {TraceSliceType::STREAM_CLOSE, "STREAM_CLOSE"}};

Trace::Trace(std::string& output_dir_path, std::shared_ptr<Performance> perf,
             bool session_enable)
    : ProfilerLifeCycle("Trace"),
      output_dir_path_(output_dir_path),
      perf_(perf),
      write_file_interval_(DEFAULT_WRITE_TRACE_INTERVAL),
      session_enable_(session_enable) {}

Trace::~Trace() {
  if (IsRunning()) {
    Stop();
  }
}

Status Trace::OnStart() {
  timer_run_ = true;
  timer_ = std::make_shared<std::thread>(&Trace::TraceWork, this);
  return STATUS_SUCCESS;
}

Status Trace::OnResume() { return OnStart(); }

Status Trace::OnStop() {
  OnPause();

  WriteTrace();
  traces_.clear();
  return STATUS_SUCCESS;
}

Status Trace::OnPause() {
  if (timer_) {
    timer_run_ = false;
    timer_->join();
    timer_ = nullptr;
  }

  return STATUS_SUCCESS;
}

std::shared_ptr<FlowUnitTrace> Trace::FlowUnit(
    const std::string& flow_unit_name) {
  std::unique_lock<std::mutex> trace_lock(trace_mutex_);

  if (traces_.find(flow_unit_name) == traces_.end()) {
    auto flow_unit_trace =
        std::shared_ptr<FlowUnitTrace>(new FlowUnitTrace(flow_unit_name));
    if (perf_ != nullptr) {
      auto flow_unit_profile = perf_->GetFlowUnitPerfCtx(flow_unit_name);
      flow_unit_trace->SetFlowUnitPerfCtx(flow_unit_profile);
    }

    traces_.insert(std::make_pair(flow_unit_name, flow_unit_trace));
    return flow_unit_trace;
  }

  return traces_[flow_unit_name];
}

std::string Trace::TraceSliceTypeToString(TraceSliceType type) {
  if (TRACE_SLICE_TYPE.find(type) == TRACE_SLICE_TYPE.end()) {
    MBLOG_ERROR << "parse TraceSliceType to string failed";
    return "";
  }

  return TRACE_SLICE_TYPE.at(type);
}

void Trace::TraceWork() {
  unsigned long now = {0};
  int32_t sleep = DEFAULT_TIMER_SAMPLE_INTERVAL;
  int32_t sleep_time = 0;
  unsigned long expect_time = 0;

  MBLOG_INFO << "trace timer start";

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
    count++;
    if (count > write_file_interval_) {
      WriteTrace();
      count = 0;
    }
  }

  MBLOG_INFO << "trace timer end";
}

Status Trace::WriteTrace() {
  nlohmann::json traces_json = nlohmann::json::array();
  std::unique_lock<std::mutex> lock(trace_mutex_);
  if (traces_.empty()) {
    return STATUS_SUCCESS;
  }

  uint64_t valid_trace_count = 0;
  for (auto trace : traces_) {
    std::string flow_unit_name = trace.second->GetFlowUnitName();
    std::vector<std::shared_ptr<TraceSlice>> trace_slices;
    trace.second->GetTraceSlices(trace_slices);
    for (auto slice : trace_slices) {
      if (slice->GetDuration() < 0) {
        continue;
      }

      valid_trace_count++;
      nlohmann::json trace_json;
      // Global
      nlohmann::json args;
      args["batch_size"] = slice->GetBatchSize();

      trace_json["name"] = TraceSliceTypeToString(slice->GetTraceSliceType());
      trace_json["dur"] = slice->GetDuration();
      trace_json["ts"] =
          slice->GetBeginEvent()->GetEventTime().time_since_epoch().count();
      trace_json["tid"] = flow_unit_name;
      trace_json["ph"] = "X";
      trace_json["pid"] = "Graph";
      trace_json["args"] = args;
      traces_json.push_back(trace_json);
      // Session
      if (session_enable_) {
        trace_json["pid"] = "Session:" + slice->GetSession();
        traces_json.push_back(trace_json);
      }
    }
  }

  lock.unlock();

  if (valid_trace_count == 0) {
    return STATUS_SUCCESS;
  }

  time_t current_time = time(nullptr);
  char buf[64] = {0};
  auto* local_tm = localtime(&current_time);
  if (local_tm) {
    strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", local_tm);
  }

  // TODO: graph_name + task_name + timestample
  std::string file_path =
      output_dir_path_ + "/" + "trace_" + std::string(buf) + ".json";

  std::ofstream out(file_path);
  if (out.is_open() == false) {
    MBLOG_ERROR << "write trace failed, file path : " << file_path;
    return STATUS_FAULT;
  }
  Defer { out.close(); };

  std::string traces_json_str = traces_json.dump();
  out.write(traces_json_str.c_str(), traces_json_str.size());
  if (out.rdstate() & std::ios::failbit) {
    MBLOG_ERROR << "Write file " << file_path << " failed";
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

FlowUnitTrace::FlowUnitTrace(const std::string& flow_unit_name)
    : flow_unit_name_(flow_unit_name) {}

std::shared_ptr<TraceSlice> FlowUnitTrace::Slice(TraceSliceType slice_type,
                                                 std::string session) {
  std::unique_lock<std::mutex> lock(trace_slices_mutex_);
  auto slice_ptr = std::shared_ptr<TraceSlice>(new TraceSlice(
      slice_type, session, shared_from_this(), flow_unit_perf_ctx_));

  return slice_ptr;
}

Status FlowUnitTrace::AddTraceSlice(std::shared_ptr<TraceSlice> trace_slice) {
  if (trace_slice == nullptr) {
    return STATUS_FAULT;
  }

  std::unique_lock<std::mutex> lock(trace_slices_mutex_);
  trace_slices_.emplace_back(trace_slice);

  return STATUS_SUCCESS;
}

void FlowUnitTrace::GetTraceSlices(
    std::vector<std::shared_ptr<TraceSlice>>& trace_slices) {
  std::unique_lock<std::mutex> lock(trace_slices_mutex_);
  trace_slices.swap(trace_slices_);
  trace_slices_.clear();
}

void FlowUnitTrace::SetFlowUnitPerfCtx(
    std::shared_ptr<FlowUnitPerfCtx> flow_unit_perf_ctx) {
  flow_unit_perf_ctx_ = flow_unit_perf_ctx;
}

TraceEvent::TraceEvent()
    : event_type_(EventType::BEGIN),
      event_time_(std::chrono::time_point_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now())),
      thread_id_(std::this_thread::get_id()) {}

TraceSlice::TraceSlice(TraceSliceType& slice_type, std::string session,
                       std::shared_ptr<FlowUnitTrace> flow_unit_trace_ptr,
                       std::shared_ptr<TraceEvent> begin,
                       std::shared_ptr<TraceEvent> end)
    : slice_type_(slice_type),
      session_(session),
      flow_unit_trace_ptr_(flow_unit_trace_ptr),
      begin_event_ptr_(begin),
      end_event_ptr_(end),
      is_end_called_(false),
      batch_size_(0) {}

TraceSlice::TraceSlice(TraceSliceType& slice_type, std::string session,
                       std::shared_ptr<FlowUnitTrace> flow_unit_trace_ptr,
                       std::shared_ptr<FlowUnitPerfCtx> flow_unit_perf_ctx)
    : slice_type_(slice_type),
      session_(session),
      flow_unit_trace_ptr_(flow_unit_trace_ptr),
      flow_unit_perf_ctx_(flow_unit_perf_ctx),
      is_end_called_(false),
      batch_size_(0) {}

TraceSlice::~TraceSlice() {
  if (!is_end_called_) {
    End();
  }
};

int32_t TraceSlice::GetDuration() {
  if (begin_event_ptr_ == nullptr || end_event_ptr_ == nullptr) {
    return -1;
  }

  std::chrono::duration<double, std::micro> duration =
      std::chrono::duration<double, std::micro>(
          end_event_ptr_->GetEventTime() - begin_event_ptr_->GetEventTime());

  return duration.count();
}

std::string TraceSlice::GetSession() { return session_; }

void TraceSlice::Begin() {
  begin_event_ptr_.reset(new TraceEvent());
  begin_event_ptr_->SetEventType(EventType::BEGIN);
  begin_event_ptr_->SetEventTime(
      std::chrono::time_point_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now()));
  begin_event_ptr_->SetThreadId(std::this_thread::get_id());
}

void TraceSlice::End() {
  is_end_called_ = true;
  auto flow_unit_trace = flow_unit_trace_ptr_.lock();
  if (flow_unit_trace == nullptr) {
    return;
  }

  end_event_ptr_.reset(new TraceEvent());
  end_event_ptr_->SetEventType(EventType::END);
  end_event_ptr_->SetEventTime(
      std::chrono::time_point_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now()));
  end_event_ptr_->SetThreadId(std::this_thread::get_id());

  std::shared_ptr<TraceSlice> new_slice_ptr(
      new TraceSlice(slice_type_, session_, flow_unit_trace, begin_event_ptr_,
                     end_event_ptr_));
  new_slice_ptr->is_end_called_ = true;
  new_slice_ptr->SetBatchSize(batch_size_);

  // FIXME : Not good to update flow unit perf in trace
  if ((TraceSliceType::PROCESS == slice_type_) &&
      (flow_unit_perf_ctx_ != nullptr)) {
    flow_unit_perf_ctx_->UpdateProcessLatency(new_slice_ptr->GetDuration());
  }

  flow_unit_trace->AddTraceSlice(new_slice_ptr);
}
}  // namespace modelbox