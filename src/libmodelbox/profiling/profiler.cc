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

#include <sys/stat.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>

namespace modelbox {

constexpr const char* PROFILE_DEFAULT_PATH = "/tmp/modelbox/perf";

ProfilerLifeCycle::ProfilerLifeCycle(std::string name)
    : name_(std::move(name)) {}

ProfilerLifeCycle::~ProfilerLifeCycle() = default;

Status ProfilerLifeCycle::Init() {
  if (is_initialized_) {
    MBLOG_INFO << name_ << " has been initialized, no need to init";
    return STATUS_SUCCESS;
  }

  auto ret = OnInit();
  if (ret != STATUS_SUCCESS) {
    return ret;
  }

  is_initialized_ = true;
  return STATUS_SUCCESS;
}

Status ProfilerLifeCycle::Start() {
  if (is_running_) {
    MBLOG_INFO << name_ << " is running, no need to start";
    return STATUS_SUCCESS;
  }

  auto ret = OnStart();
  if (ret != STATUS_SUCCESS) {
    return ret;
  }

  is_running_ = true;
  return STATUS_SUCCESS;
}

Status ProfilerLifeCycle::Stop() {
  if (!is_running_) {
    return STATUS_SUCCESS;
  }

  auto ret = OnStop();
  if (ret != STATUS_SUCCESS) {
    return ret;
  }

  is_running_ = false;
  return STATUS_SUCCESS;
}

Status ProfilerLifeCycle::Pause() {
  if (!is_running_) {
    return STATUS_SUCCESS;
  }

  auto ret = OnPause();
  if (ret != STATUS_SUCCESS) {
    return ret;
  }

  is_running_ = false;
  return STATUS_SUCCESS;
}

Status ProfilerLifeCycle::Resume() {
  if (is_running_) {
    MBLOG_INFO << name_ << " is running, no need to resume";
    return STATUS_SUCCESS;
  }

  auto ret = OnResume();
  if (ret != STATUS_SUCCESS) {
    return ret;
  }

  is_running_ = true;
  return STATUS_SUCCESS;
}

Profiler::Profiler(std::shared_ptr<DeviceManager> device_mgr,
                   std::shared_ptr<Configuration> config)
    : ProfilerLifeCycle("Profiler"),
      device_mgr_(std::move(device_mgr)),
      config_(std::move(config)),
      perf_(nullptr),
      trace_(nullptr) {}

Profiler::~Profiler() {
  if (IsRunning()) {
    Stop();
  }
}

Status Profiler::OnInit() {
  bool profile_enable = false;
  bool trace_enable = false;
  bool session_enable = false;

  profile_enable = config_->GetBool("profile.profile");
  trace_enable = config_->GetBool("profile.trace");
  session_enable = config_->GetBool("profile.session");

  if (profile_enable || trace_enable) {
    auto ret = InitProfilerDir();
    if (ret != STATUS_OK) {
      return STATUS_FAULT;
    }
  }

  if (profile_enable) {
    perf_ = std::make_shared<Performance>(device_mgr_, output_dir_path_);
    perf_->Init();
  }

  if (trace_enable) {
    trace_ = std::make_shared<Trace>(output_dir_path_, perf_, session_enable);
  }

  return STATUS_SUCCESS;
}

Status Profiler::InitProfilerDir() {
  auto* profile_dir_path = getenv(PROFILE_PATH_ENV);
  if (profile_dir_path == nullptr) {
    output_dir_path_ = config_->GetString("profile.dir", PROFILE_DEFAULT_PATH);
  } else {
    output_dir_path_ = profile_dir_path;
  }

  const std::string filter_dir =
      "(/bin)|(/boot)|(/sbin)|(/etc)|(/dev)|(/proc)|(/sys)|(/var)";
  const std::string black_dir_str =
      filter_dir + "|" + "((" + filter_dir + ")/.*)";
  std::regex valid_str(black_dir_str);

  output_dir_path_ = PathCanonicalize(output_dir_path_);
  if (std::regex_match(output_dir_path_, valid_str)) {
    MBLOG_ERROR << "profiler dir invalid, please type valid profiler dir.";
    return STATUS_FAULT;
  }

  if (output_dir_path_.length() <= 0) {
    output_dir_path_ = PROFILE_DEFAULT_PATH;
  }

  MBLOG_INFO << "profiler save dir: " << output_dir_path_;
  Status ret = CreateDirectory(output_dir_path_);
  if (ret != STATUS_OK) {
    MBLOG_FATAL << "create directory : " << output_dir_path_ << " failed, "
                << ret;
    return ret;
  }

  return STATUS_SUCCESS;
}

Status Profiler::OnStart() {
  if (perf_ != nullptr) {
    perf_->Start();
  }

  if (trace_ != nullptr) {
    trace_->Start();
  }

  return STATUS_SUCCESS;
}

Status Profiler::OnStop() {
  if (perf_ != nullptr) {
    perf_->Stop();
  }

  if (trace_ != nullptr) {
    trace_->Stop();
  }

  return STATUS_SUCCESS;
}

Status Profiler::OnResume() {
  if (perf_ != nullptr) {
    perf_->Resume();
  }

  if (trace_ != nullptr) {
    trace_->Resume();
  }

  return STATUS_SUCCESS;
}

Status Profiler::OnPause() {
  if (perf_ != nullptr) {
    perf_->Pause();
  }

  if (trace_ != nullptr) {
    trace_->Pause();
  }

  return STATUS_SUCCESS;
}

}  // namespace modelbox