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

#include <modelbox/server/task.h>
#include <modelbox/server/task_manager.h>

#include <condition_variable>
#include <fstream>
#include <mutex>

#include "gtest/gtest.h"
#include "mock_driver_ctl.h"
#include "mockflow.h"
#include "modelbox/base/log.h"

namespace modelbox {

std::condition_variable cv;
std::mutex mtx;
int count;
class TaskManagerTest : public testing::Test {
 public:
  TaskManagerTest() = default;

 protected:
  void SetUp() override {
    std::string toml_content = R"(
    [driver]
    skip-default=true
    dir=[")" + std::string(TEST_LIB_DIR) +
                               "\"]\n    " +
                               R"(
    [graph]
    graphconf = '''digraph demo {
          input1[type=input, device=cpu,deviceid=0] 
          output1[type=output, device=cpu, deviceid=0]
          stream_start[type=flowunit, flowunit=virtual_stream_start, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          stream_mid[type=flowunit, flowunit=virtual_stream_mid, device=cpu, deviceid=0, label="<In_1> | <Out_1>"]
          
          input1 ->stream_start:In_1
          stream_start:Out_1 ->stream_mid:In_1
          stream_mid:Out_1->output1

        }'''
    format = "graphviz"
  )";
    mockflow_ = std::make_shared<MockFlow>();
    mockflow_->Init();
    auto ret = mockflow_->BuildAndRun("TaskManager", toml_content, -1);
  };
  void TearDown() override { mockflow_ = nullptr; };

  std::shared_ptr<MockFlow> mockflow_;

 private:
};

void TaskFinished(OneShotTask *task, TaskStatus status) {
  count++;
  EXPECT_EQ(FINISHED, status);
  cv.notify_one();
}

void TaskStopped(OneShotTask *task, TaskStatus status) {
  EXPECT_EQ(STOPPED, status);
  cv.notify_one();
}

void TaskEnd(OneShotTask *task, TaskStatus status) {
  MBLOG_INFO << "Task end " << status;
}

TEST_F(TaskManagerTest, CreateTask) {
  std::unique_lock<std::mutex> lck(mtx);
  auto tm = std::make_shared<TaskManager>(mockflow_->GetFlow(), 10);
  auto status = tm->Start();
  EXPECT_EQ(status, STATUS_SUCCESS);
  auto task_1 =
      std::dynamic_pointer_cast<OneShotTask>(tm->CreateTask(TASK_ONESHOT));
  EXPECT_EQ(WAITING, task_1->GetTaskStatus());
  auto output_buf = task_1->CreateBufferList();
  output_buf->Build({3 * sizeof(int)});
  auto *data = (int *)output_buf->MutableData();
  data[0] = 0;
  data[1] = 30000;
  data[2] = 3;
  std::unordered_map<std::string, std::shared_ptr<BufferList>> datas;
  datas.emplace("input1", output_buf);
  task_1->FillData(datas);
  task_1->RegisterStatusCallback(TaskFinished);
  task_1->Start();
  cv.wait_for(lck, std::chrono::seconds(10),
              [task_1]() { return task_1->GetTaskStatus() == FINISHED; });
  EXPECT_EQ(FINISHED, task_1->GetTaskStatus());

  auto task_2 =
      std::dynamic_pointer_cast<OneShotTask>(tm->CreateTask(TASK_ONESHOT));
  EXPECT_EQ(WAITING, task_2->GetTaskStatus());
  task_2->Start();
  task_2->RegisterStatusCallback(TaskStopped);
  cv.wait_for(lck, std::chrono::seconds(10),
              [task_2]() { return task_2->GetTaskStatus() == STOPPED; });
  EXPECT_EQ(STOPPED, task_2->GetTaskStatus());
}

TEST_F(TaskManagerTest, StopTask) {
  std::unique_lock<std::mutex> lck(mtx);
  auto tm = std::make_shared<TaskManager>(mockflow_->GetFlow(), 10);
  auto status = tm->Start();
  EXPECT_EQ(status, STATUS_SUCCESS);
  auto task_1 =
      std::dynamic_pointer_cast<OneShotTask>(tm->CreateTask(TASK_ONESHOT));
  EXPECT_EQ(WAITING, task_1->GetTaskStatus());
  auto output_buf = task_1->CreateBufferList();
  output_buf->Build({3 * sizeof(int)});
  auto *data = (int *)output_buf->MutableData();
  data[0] = 0;
  data[1] = 30000;
  data[2] = 3;
  std::unordered_map<std::string, std::shared_ptr<BufferList>> datas;
  datas.emplace("input1", output_buf);
  task_1->FillData(datas);
  task_1->RegisterStatusCallback(TaskStopped);
  task_1->Start();
  task_1->Stop();
  EXPECT_EQ(STOPPED, task_1->GetTaskStatus());

  auto task_2 =
      std::dynamic_pointer_cast<OneShotTask>(tm->CreateTask(TASK_ONESHOT));
  EXPECT_EQ(WAITING, task_2->GetTaskStatus());
  task_2->RegisterStatusCallback(TaskStopped);
  task_2->Stop();
  EXPECT_EQ(STOPPED, task_2->GetTaskStatus());

  auto task_3 =
      std::dynamic_pointer_cast<OneShotTask>(tm->CreateTask(TASK_ONESHOT));
  EXPECT_EQ(WAITING, task_3->GetTaskStatus());
  task_3->Start();
  task_3->Stop();
  EXPECT_EQ(STOPPED, task_3->GetTaskStatus());

  EXPECT_EQ(tm->GetAvaiableTaskCount(), 0);
}

TEST_F(TaskManagerTest, DeleteTaskById) {
  std::unique_lock<std::mutex> lck(mtx);
  auto tm = std::make_shared<TaskManager>(mockflow_->GetFlow(), 10);
  auto status = tm->Start();
  EXPECT_EQ(status, STATUS_SUCCESS);

  auto task =
      std::dynamic_pointer_cast<OneShotTask>(tm->CreateTask(TASK_ONESHOT));
  auto uuid = task->GetTaskId();
  EXPECT_EQ(WAITING, task->GetTaskStatus());
  auto output_buf = task->CreateBufferList();
  output_buf->Build({3 * sizeof(int)});
  auto *data = (int *)output_buf->MutableData();
  data[0] = 0;
  data[1] = 30000;
  data[2] = 3;
  std::unordered_map<std::string, std::shared_ptr<BufferList>> datas;
  datas.emplace("input1", output_buf);
  task->FillData(datas);
  task->RegisterStatusCallback(TaskEnd);
  task->Start();
  sleep(1);
  auto get_task = tm->GetTaskById(uuid);
  EXPECT_TRUE(get_task != nullptr);
  sleep(1);
  tm->DeleteTaskById(uuid);
  auto del_task = tm->GetTaskById(uuid);
  EXPECT_TRUE(del_task == nullptr);
}

TEST_F(TaskManagerTest, TaskInQueue) {
  count = 0;
  std::unique_lock<std::mutex> lck(mtx);
  auto tm = std::make_shared<TaskManager>(mockflow_->GetFlow(), 3);
  auto status = tm->Start();
  EXPECT_EQ(status, STATUS_SUCCESS);

  for (uint32_t i = 0; i < 4; i++) {
    auto task =
        std::dynamic_pointer_cast<OneShotTask>(tm->CreateTask(TASK_ONESHOT));
    auto uuid = task->GetTaskId();
    EXPECT_EQ(WAITING, task->GetTaskStatus());
    auto output_buf = task->CreateBufferList();
    output_buf->Build({3 * sizeof(int)});
    auto *data = (int *)output_buf->MutableData();
    data[0] = 0;
    data[1] = 40000;
    data[2] = 3;
    std::unordered_map<std::string, std::shared_ptr<BufferList>> datas;
    datas.emplace("input1", output_buf);
    task->FillData(datas);
    task->Start();
  }
  sleep(1);
  int running_tasks = 0;
  int waitting_tasks = 0;
  int stopped_tasks = 0;
  int finish_tasks = 0;

  auto task_list = tm->GetAllTasks();

  std::shared_ptr<OneShotTask> running_task;

  for (const auto &task : task_list) {
    auto one_shot_task = std::dynamic_pointer_cast<OneShotTask>(task);
    one_shot_task->RegisterStatusCallback(TaskFinished);
    if (task->GetTaskStatus() == WORKING) {
      running_task = one_shot_task;
      running_tasks++;
    }
    if (task->GetTaskStatus() == WAITING) {
      waitting_tasks++;
    }
    if (task->GetTaskStatus() == STOPPED) {
      stopped_tasks++;
    }
    if (task->GetTaskStatus() == FINISHED) {
      finish_tasks++;
    }
  }
  EXPECT_EQ(running_tasks, 3);
  EXPECT_EQ(waitting_tasks, 1);
  EXPECT_EQ(stopped_tasks, 0);
  EXPECT_EQ(finish_tasks, 0);

  running_task->RegisterStatusCallback(TaskStopped);
  running_task->Stop();

  cv.wait(lck, []() { return count >= 3; });
  running_tasks = 0;
  waitting_tasks = 0;
  stopped_tasks = 0;
  finish_tasks = 0;
  for (const auto &task : task_list) {
    if (task->GetTaskStatus() == WORKING) {
      running_tasks++;
    }
    if (task->GetTaskStatus() == WAITING) {
      waitting_tasks++;
    }
    if (task->GetTaskStatus() == STOPPED) {
      stopped_tasks++;
    }
    if (task->GetTaskStatus() == FINISHED) {
      finish_tasks++;
    }
  }
  EXPECT_EQ(running_tasks, 0);
  EXPECT_EQ(waitting_tasks, 0);
  EXPECT_EQ(stopped_tasks, 1);
  EXPECT_EQ(finish_tasks, 3);
}

}  // namespace modelbox