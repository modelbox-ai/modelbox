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

#include <modelbox/base/any.h>
#include <dirent.h>
#include <sys/stat.h>

#include <atomic>

#include "modelbox/statistics.h"
#include "gtest/gtest.h"
#include "test_config.h"

class ProfilerTest : public testing::Test {
 public:
  ProfilerTest() {}
  virtual ~ProfilerTest() {}

 protected:
  virtual void SetUp() {
    std::ostringstream sstr_profile_path;
    sstr_profile_path << TEST_DATA_DIR << "/perf";
    setenv(modelbox::PROFILE_PATH_ENV, sstr_profile_path.str().c_str(), 1);
  };

  virtual void TearDown() { unsetenv(modelbox::PROFILE_PATH_ENV); };
};

TEST_F(ProfilerTest, ProfilerInit) {
  auto device_manager = std::make_shared<modelbox::DeviceManager>();
  auto config = std::make_shared<modelbox::Configuration>();
  auto profiler = std::make_shared<modelbox::Profiler>(device_manager, config);
  EXPECT_EQ(profiler->Init(), modelbox::STATUS_SUCCESS);
  EXPECT_EQ(profiler->Init(), modelbox::STATUS_SUCCESS);
  EXPECT_TRUE(profiler->IsInitialized());
}

TEST_F(ProfilerTest, ProfilerStartAndStop) {
  auto device_manager = std::make_shared<modelbox::DeviceManager>();
  auto config = std::make_shared<modelbox::Configuration>();
  auto profiler = std::make_shared<modelbox::Profiler>(device_manager, config);
  EXPECT_EQ(profiler->Init(), modelbox::STATUS_SUCCESS);
  EXPECT_EQ(profiler->Start(), modelbox::STATUS_SUCCESS);
  EXPECT_EQ(profiler->Start(), modelbox::STATUS_SUCCESS);
  EXPECT_TRUE(profiler->IsRunning());
  EXPECT_EQ(profiler->Stop(), modelbox::STATUS_SUCCESS);
  EXPECT_FALSE(profiler->IsRunning());
}

TEST_F(ProfilerTest, ProfilerPauseAndResume) {
  auto device_manager = std::make_shared<modelbox::DeviceManager>();
  auto config = std::make_shared<modelbox::Configuration>();
  auto profiler = std::make_shared<modelbox::Profiler>(device_manager, config);
  EXPECT_EQ(profiler->Init(), modelbox::STATUS_SUCCESS);
  EXPECT_EQ(profiler->Start(), modelbox::STATUS_SUCCESS);
  EXPECT_EQ(profiler->Pause(), modelbox::STATUS_SUCCESS);
  EXPECT_FALSE(profiler->IsRunning());
  EXPECT_EQ(profiler->Resume(), modelbox::STATUS_SUCCESS);
  EXPECT_TRUE(profiler->IsRunning());
}

TEST_F(ProfilerTest, ProfilerSetTraceSlice) {
  auto device_manager = std::make_shared<modelbox::DeviceManager>();
  auto config = std::make_shared<modelbox::Configuration>();
  config->SetProperty("profile.trace", "true");
  auto profiler = std::make_shared<modelbox::Profiler>(device_manager, config);
  profiler->Init();
  std::string session = "session_0";
  auto trace = profiler->GetTrace();
  {
    auto trace_slice = trace->FlowUnit("test")->Slice(
        modelbox::TraceSliceType::PROCESS, session);
    trace_slice->Begin();
    trace_slice->End();
  }

  std::vector<std::shared_ptr<modelbox::TraceSlice>> all_slices;
  trace->FlowUnit("test")->GetTraceSlices(all_slices);
  int process_slices = 0;
  std::shared_ptr<modelbox::TraceSlice> last_slice;
  for (auto slice : all_slices) {
    if (slice->GetTraceSliceType() == modelbox::TraceSliceType::PROCESS) {
      process_slices++;
      last_slice = slice;
    }
  }

  EXPECT_EQ(process_slices, 1);
  EXPECT_NE(last_slice->GetBeginEvent(), nullptr);
  EXPECT_NE(last_slice->GetEndEvent(), nullptr);
}

TEST_F(ProfilerTest, ProfilerTraceSliceEndNotCalled) {
  auto device_manager = std::make_shared<modelbox::DeviceManager>();
  auto config = std::make_shared<modelbox::Configuration>();
  config->SetProperty("profile.trace", "true");
  auto profiler = std::make_shared<modelbox::Profiler>(device_manager, config);
  std::string session = "session_0";
  profiler->Init();
  auto trace = profiler->GetTrace();
  {
    auto trace_slice = trace->FlowUnit("test")->Slice(
        modelbox::TraceSliceType::PROCESS, session);
    trace_slice->Begin();
  }

  std::vector<std::shared_ptr<modelbox::TraceSlice>> all_slices;
  trace->FlowUnit("test")->GetTraceSlices(all_slices);

  int process_slices = 0;
  std::shared_ptr<modelbox::TraceEvent> last_event;
  for (auto slice : all_slices) {
    if (slice->GetTraceSliceType() == modelbox::TraceSliceType::PROCESS) {
      process_slices++;
      last_event = slice->GetEndEvent();
    }
  }

  EXPECT_EQ(process_slices, 1);
  EXPECT_NE(last_event.get(), nullptr);
}

TEST_F(ProfilerTest, ProfilerTimer) {
  auto device_manager = std::make_shared<modelbox::DeviceManager>();
  auto config = std::make_shared<modelbox::Configuration>();
  auto profiler = std::make_shared<modelbox::Profiler>(device_manager, config);
  profiler->Init();
  auto perf = profiler->GetPerf();
  EXPECT_EQ(profiler->Init(), modelbox::STATUS_SUCCESS);
  EXPECT_EQ(profiler->Start(), modelbox::STATUS_SUCCESS);
}

TEST_F(ProfilerTest, TracePerf) {
  auto deviceManager = std::make_shared<modelbox::DeviceManager>();
  auto config = std::make_shared<modelbox::Configuration>();
  config->SetProperty("profile.trace", "true");
  std::shared_ptr<modelbox::Profiler> profiler =
      std::make_shared<modelbox::Profiler>(deviceManager, config);
  profiler->Init();
  auto trace = profiler->GetTrace();

  profiler->Start();

  auto flow_unit_test = trace->FlowUnit("resize");
  for (int i = 0; i < 1000; i++) {
    auto open_slice =
        flow_unit_test->Slice(modelbox::TraceSliceType::OPEN, "session");
    open_slice->Begin();
    open_slice->End();
  }
}

TEST_F(ProfilerTest, WriteTrace) {
  MBLOG_INFO << "PROFILE_PATH : " << getenv(modelbox::PROFILE_PATH_ENV);
  auto deviceManager = std::make_shared<modelbox::DeviceManager>();
  auto config = std::make_shared<modelbox::Configuration>();
  config->SetProperty("profile.trace", "true");
  std::shared_ptr<modelbox::Profiler> profiler =
      std::make_shared<modelbox::Profiler>(deviceManager, config);
  profiler->Init();
  auto trace = profiler->GetTrace();
  auto flow_unit_trace_resize = trace->FlowUnit("resize");
  auto flow_unit_trace_crop = trace->FlowUnit("crop");
  auto flow_unit_trace_preprocess = trace->FlowUnit("preprocess");
  auto flow_unit_trace_infer = trace->FlowUnit("infer");

  std::string session = "session_0";
  // OPEN
  std::thread process_resize_open([&]() {
    auto open_slice =
        flow_unit_trace_resize->Slice(modelbox::TraceSliceType::OPEN, session);
    open_slice->Begin();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    open_slice->End();
  });

  std::thread process_crop_open([&]() {
    auto open_slice =
        flow_unit_trace_crop->Slice(modelbox::TraceSliceType::OPEN, session);
    open_slice->Begin();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    open_slice->End();
  });

  std::thread process_preprocess_open([&]() {
    auto open_slice = flow_unit_trace_preprocess->Slice(
        modelbox::TraceSliceType::OPEN, session);
    open_slice->Begin();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    open_slice->End();
  });

  std::thread process_infer_open([&]() {
    auto open_slice =
        flow_unit_trace_infer->Slice(modelbox::TraceSliceType::OPEN, session);
    open_slice->Begin();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    open_slice->End();
  });

  process_resize_open.join();
  process_infer_open.join();
  process_preprocess_open.join();
  process_crop_open.join();

  // PROCESS
  std::thread process_resize([&]() {
    for (int i = 0; i < 5; i++) {
      auto process_slice = flow_unit_trace_resize->Slice(
          modelbox::TraceSliceType::PROCESS, session);
      process_slice->Begin();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      process_slice->End();
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  std::thread process_crop([&]() {
    for (int i = 0; i < 5; i++) {
      auto process_slice =
          flow_unit_trace_crop->Slice(modelbox::TraceSliceType::PROCESS, session);
      process_slice->Begin();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      process_slice->End();
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  std::thread process_preprocess([&]() {
    for (int i = 0; i < 5; i++) {
      auto process_slice = flow_unit_trace_preprocess->Slice(
          modelbox::TraceSliceType::PROCESS, session);
      process_slice->Begin();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      process_slice->End();
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  // ANOTHER
  std::thread process_infer([&]() {
    for (int i = 0; i < 5; i++) {
      auto process_slice = flow_unit_trace_infer->Slice(
          modelbox::TraceSliceType::PROCESS, session);
      process_slice->Begin();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      process_slice->End();
    }
  });

  process_resize.join();
  process_crop.join();
  process_preprocess.join();
  process_infer.join();

  // CLOSE
  std::thread process_resize_close([&]() {
    auto close_slice =
        flow_unit_trace_resize->Slice(modelbox::TraceSliceType::CLOSE, session);
    close_slice->Begin();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    close_slice->End();
  });

  std::thread process_crop_close([&]() {
    auto close_slice =
        flow_unit_trace_crop->Slice(modelbox::TraceSliceType::CLOSE, session);
    close_slice->Begin();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    close_slice->End();
  });

  std::thread process_preprocess_close([&]() {
    auto close_slice = flow_unit_trace_preprocess->Slice(
        modelbox::TraceSliceType::CLOSE, session);
    close_slice->Begin();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    close_slice->End();
  });

  std::thread process_infer_close([&]() {
    auto close_slice =
        flow_unit_trace_infer->Slice(modelbox::TraceSliceType::CLOSE, session);
    close_slice->Begin();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    close_slice->End();
  });

  process_resize_close.join();
  process_infer_close.join();
  process_preprocess_close.join();
  process_crop_close.join();

  modelbox::Status ret = trace->WriteTrace();
  EXPECT_EQ(ret, modelbox::STATUS_SUCCESS);
}

TEST_F(ProfilerTest, FlowUnitProfile) {
  auto deviceManager = std::make_shared<modelbox::DeviceManager>();
  auto config = std::make_shared<modelbox::Configuration>();
  config->SetProperty("profile.profile", "true");
  std::shared_ptr<modelbox::Profiler> profiler =
      std::make_shared<modelbox::Profiler>(deviceManager, config);

  auto flow_unit_perf_ctx = std::make_shared<modelbox::FlowUnitPerfCtx>("resize");

  flow_unit_perf_ctx->UpdateProcessLatency(10);
  flow_unit_perf_ctx->UpdateProcessLatency(10);
  flow_unit_perf_ctx->UpdateProcessLatency(1);
  int32_t process_latency = flow_unit_perf_ctx->GetProcessLatency();
  EXPECT_EQ(process_latency, 7);

  std::string device_type_1 = "GPU";
  std::string device_id_1 = "001";

  flow_unit_perf_ctx->UpdateDeviceMemory(device_type_1, device_id_1, 20);
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  flow_unit_perf_ctx->UpdateDeviceMemory(device_type_1, device_id_1, 30);
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  flow_unit_perf_ctx->UpdateDeviceMemory(device_type_1, device_id_1, 50);
  int32_t device_memory =
      flow_unit_perf_ctx->GetDeviceMemory(device_type_1, device_id_1);
  EXPECT_EQ(device_memory, 50);
}

TEST_F(ProfilerTest, Statistics) {
  std::atomic<size_t> create_notify_count(0);
  std::atomic<size_t> delete_notify_count(0);
  std::atomic<size_t> change_notify_count(0);
  std::atomic<size_t> timer_notify_count(0);
  {
    auto root = std::make_shared<modelbox::StatisticsItem>();
    const std::string path_pattern = "flow.*.VideoDecoder.frame_count";
    const std::string frame_key = "frame_count";
    std::set<std::string> expect_val;
    // Plugin register notify
    auto create_notify_cfg = std::make_shared<modelbox::StatisticsNotifyCfg>(
        path_pattern,
        [&create_notify_count](
            const std::shared_ptr<const modelbox::StatisticsNotifyMsg>& msg) {
          MBLOG_INFO << "Create notify [" << msg->path_ << "]";
          EXPECT_EQ(msg->type_, modelbox::StatisticsNotifyType::CREATE);
          EXPECT_EQ(msg->path_, "flow.SessionId.VideoDecoder.frame_count");
          EXPECT_TRUE(msg->value_->IsUint64());
          EXPECT_FALSE(msg->value_->IsString());
          uint64_t frame_count = 0;
          auto ret = msg->value_->GetUint64(frame_count);
          EXPECT_TRUE(ret);
          EXPECT_EQ(frame_count, 0);
          ++create_notify_count;
        },
        modelbox::StatisticsNotifyType::CREATE);
    root->RegisterNotify(create_notify_cfg);

    auto delete_notify_cfg = std::make_shared<modelbox::StatisticsNotifyCfg>(
        path_pattern,
        [&delete_notify_count](
            const std::shared_ptr<const modelbox::StatisticsNotifyMsg>& msg) {
          MBLOG_INFO << "Delete notify [" << msg->path_ << "]";
          EXPECT_EQ(msg->type_, modelbox::StatisticsNotifyType::DELETE);
          EXPECT_EQ(msg->path_, "flow.SessionId.VideoDecoder.frame_count");
          EXPECT_TRUE(msg->value_->IsUint64());
          EXPECT_FALSE(msg->value_->IsString());
          uint64_t frame_count = 0;
          auto ret = msg->value_->GetUint64(frame_count);
          EXPECT_TRUE(ret);
          EXPECT_EQ(frame_count, 1);
          ++delete_notify_count;
        },
        modelbox::StatisticsNotifyType::DELETE);
    root->RegisterNotify(delete_notify_cfg);

    auto change_notify_cfg = std::make_shared<modelbox::StatisticsNotifyCfg>(
        path_pattern,
        [&change_notify_count](
            const std::shared_ptr<const modelbox::StatisticsNotifyMsg>& msg) {
          MBLOG_INFO << "Change notify [" << msg->path_ << "]";
          EXPECT_EQ(msg->type_, modelbox::StatisticsNotifyType::CHANGE);
          EXPECT_EQ(msg->path_, "flow.SessionId.VideoDecoder.frame_count");
          EXPECT_TRUE(msg->value_->IsUint64());
          EXPECT_FALSE(msg->value_->IsString());
          uint64_t frame_count = 0;
          auto ret = msg->value_->GetUint64(frame_count);
          EXPECT_TRUE(ret);
          EXPECT_EQ(frame_count, 1);
          ++change_notify_count;
        },
        modelbox::StatisticsNotifyType::CHANGE);
    root->RegisterNotify(change_notify_cfg);

    auto timer_notify_cfg = std::make_shared<modelbox::StatisticsNotifyCfg>(
        path_pattern,
        [&timer_notify_count](
            const std::shared_ptr<const modelbox::StatisticsNotifyMsg>& msg) {
          MBLOG_INFO << "Timer notify [" << msg->path_ << "]";
          EXPECT_EQ(msg->type_, modelbox::StatisticsNotifyType::TIMER);
          EXPECT_EQ(msg->path_, "flow.SessionId.VideoDecoder.frame_count");
          EXPECT_TRUE(msg->value_->IsUint64());
          EXPECT_FALSE(msg->value_->IsString());
          uint64_t frame_count = 0;
          auto ret = msg->value_->GetUint64(frame_count);
          EXPECT_TRUE(ret);
          EXPECT_EQ(frame_count, 1);
          ++timer_notify_count;
        });
    timer_notify_cfg->SetNotifyTimer(100, 100);
    root->RegisterNotify(timer_notify_cfg);
    // FlowUnit
    auto flow_item = root->AddItem(modelbox::STATISTICS_ITEM_FLOW);
    auto session_item = flow_item->AddItem("SessionId");
    auto decoder_item = session_item->AddItem("VideoDecoder");
    // Device
    auto device_item = root->AddItem("Device");
    auto gpu0_item = device_item->AddItem("gpu0");
    auto gpu1_item = device_item->AddItem("gpu1", std::string(""));
    // Check item
    expect_val = {modelbox::STATISTICS_ITEM_FLOW, "Device"};
    EXPECT_EQ(root->GetItemNames(), expect_val);
    expect_val = {"SessionId"};
    EXPECT_EQ(flow_item->GetItemNames(), expect_val);
    expect_val = {"VideoDecoder"};
    EXPECT_EQ(session_item->GetItemNames(), expect_val);
    EXPECT_EQ(decoder_item->GetName(), "VideoDecoder");
    EXPECT_EQ(decoder_item->GetPath(), "flow.SessionId.VideoDecoder");

    expect_val = {"gpu0", "gpu1"};
    EXPECT_EQ(device_item->GetItemNames(), expect_val);
    EXPECT_EQ(gpu0_item->GetName(), "gpu0");
    EXPECT_EQ(gpu0_item->GetPath(), "Device.gpu0");
    EXPECT_EQ(gpu1_item->GetName(), "gpu1");
    EXPECT_EQ(gpu1_item->GetPath(), "Device.gpu1");

    // Add frame count
    uint64_t init_frame_count = 0;
    auto frame_item = decoder_item->AddItem(frame_key, init_frame_count);

    // Wrong type
    auto ret = frame_item->IncreaseValue<uint32_t>(1);
    EXPECT_EQ(ret, modelbox::STATUS_INVALID);

    uint32_t wrong_type_frame_count;
    ret = frame_item->GetValue(wrong_type_frame_count);
    EXPECT_NE(ret, modelbox::STATUS_OK);

    // Right op
    std::this_thread::sleep_for(
        std::chrono::seconds(1));  // Wait change notify cool down
    ret = frame_item->IncreaseValue<uint64_t>(1);
    EXPECT_EQ(ret, modelbox::STATUS_SUCCESS);
    ret = frame_item->IncreaseValue<uint64_t>(1);
    EXPECT_EQ(ret, modelbox::STATUS_SUCCESS);

    std::this_thread::sleep_for(
        std::chrono::seconds(1));  // Wait change notify cool down
    ret = frame_item->SetValue<uint64_t>(1);
    EXPECT_EQ(ret, modelbox::STATUS_SUCCESS);
    ret = frame_item->SetValue<uint64_t>(1);
    EXPECT_EQ(ret, modelbox::STATUS_SUCCESS);

    uint64_t frame_count = 0;
    ret = frame_item->GetValue(frame_count);
    EXPECT_EQ(ret, modelbox::STATUS_OK);
    EXPECT_EQ(frame_count, 1);

    ret = gpu0_item->SetValue<uint64_t>(1);
    EXPECT_EQ(ret, modelbox::STATUS_NOTSUPPORT);
    // Foreach
    std::atomic_size_t foreach_count(0);
    root->ForEach(
        [&foreach_count](const std::shared_ptr<modelbox::StatisticsItem>& item,
                         const std::string& relative_path) {
          auto value = item->GetValue();
          MBLOG_INFO << "Foreach : " << item->GetPath() << " : "
                     << (value ? value->ToString() : "null");
          EXPECT_EQ(relative_path, item->GetName());
          ++foreach_count;
          return modelbox::STATUS_OK;
        });
    EXPECT_EQ(foreach_count, 2);
    foreach_count = 0;
    root->ForEach(
        [&foreach_count](const std::shared_ptr<modelbox::StatisticsItem>& item,
                         const std::string& relative_path) {
          auto value = item->GetValue();
          MBLOG_INFO << "Foreach : " << item->GetPath() << " : "
                     << (value ? value->ToString() : "null");
          ++foreach_count;
          return modelbox::STATUS_OK;
        },
        true);
    EXPECT_EQ(foreach_count, 7);
    decoder_item->ForEach(
        [](const std::shared_ptr<modelbox::StatisticsItem>& item,
           const std::string& relative_path) {
          auto value = item->GetValue();
          MBLOG_INFO << "Foreach : " << item->GetPath() << " : "
                     << (value ? value->ToString() : "null") << " : "
                     << relative_path;
          EXPECT_EQ(item->GetName(), relative_path);
          return modelbox::STATUS_OK;
        });
    // Read
    auto item = root->GetItem("flow");
    ASSERT_NE(item, nullptr);
    EXPECT_EQ(item->GetName(), "flow");
    item = root->GetItem("flow.SessionId.VideoDecoder");
    ASSERT_NE(item, nullptr);
    EXPECT_EQ(item->GetName(), "VideoDecoder");
    // Trigger timer
    frame_item->Notify(modelbox::StatisticsNotifyType::TIMER);
    // Unregister notify
    root->UnRegisterNotify(timer_notify_cfg);
    frame_item->Notify(
        modelbox::StatisticsNotifyType::TIMER);  // this should not working
    // Remove ctx
    decoder_item->Dispose();
    EXPECT_TRUE(session_item->GetItemNames().empty());
    EXPECT_EQ(decoder_item->AddItem("test"), nullptr);
    EXPECT_TRUE(decoder_item->GetItemNames().empty());
    flow_item->DelItem("SessionId");
    EXPECT_TRUE(flow_item->GetItemNames().empty());
    root->DelItem(modelbox::STATISTICS_ITEM_FLOW);
    expect_val = {"Device"};
    EXPECT_EQ(root->GetItemNames(), expect_val);

    std::string val;
    gpu1_item->SetValue(std::string("test"));
    gpu1_item->GetValue(val);
    EXPECT_EQ(val, "test");
    gpu1_item->SetValue(std::string("test2"));
    gpu1_item->GetValue(val);
    EXPECT_EQ(val, "test2");
    // Destroy
  }
  // Check callback
  EXPECT_EQ(create_notify_count, 1);
  EXPECT_EQ(change_notify_count, 2);
  EXPECT_EQ(delete_notify_count, 1);
  EXPECT_EQ(timer_notify_count, 1);
}
