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


#ifndef MODELBOX_FLOWUNIT_ASCEND_VIDEODECODE_H_
#define MODELBOX_FLOWUNIT_ASCEND_VIDEODECODE_H_

#include <acl/acl.h>
#include <modelbox/base/blocking_queue.h>
#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>
#include <modelbox/data_context.h>

#include <string>
#include <vector>

#include "acl/ops/acl_dvpp.h"

constexpr const int ACL_PROCESS_WAIT_TIME_OUT_MS = 50;
constexpr const uint32_t YUV_BYTES_NU = 3;
constexpr const uint32_t YUV_BYTES_DE = 2;
constexpr const char *INSTANCE_ID = "instance_id";
constexpr const char *OUTPUT_PIX_FMT = "nv12";

class ThreadHandler {
 public:
  ThreadHandler(int device_id, int instance_id,
                std::shared_ptr<modelbox::DataContext> data_ctx);
  virtual ~ThreadHandler();

  modelbox::Status CreateThread();
  pthread_t GetThreadId() { return threadId_; }
  void UpdateNeedStop();

 private:
  static void *ThreadFunc(void *arg);
  pthread_t threadId_{0};
  int device_id_;
  int instance_id_;
  std::weak_ptr<modelbox::DataContext> data_ctx_;
  std::atomic<bool> need_stop_{false};
};

class DvppPacket {
 public:
  DvppPacket(size_t size, int32_t width, int32_t height, int32_t pts)
      : size_(size), width_(width), height_(height), pts_(pts){};

  DvppPacket() { stream_desc_ = nullptr; };

  virtual ~DvppPacket() = default;

  acldvppStreamDesc *GetStreamDesc() { return stream_desc_; };
  void SetStreamDesc(acldvppStreamDesc *stream_desc) {
    stream_desc_ = stream_desc;
  }

  int32_t GetPts() { return pts_; }

  int32_t GetWidth() { return width_; };
  int32_t GetHeight() { return height_; };
  bool IsEnd() { return is_end_; };
  void SetEnd(bool is_end) { is_end_ = is_end; };

 private:
  uint32_t size_{0};
  int32_t width_{0};
  int32_t height_{0};
  int32_t pts_{0};
  bool is_end_{false};
  acldvppStreamDesc *stream_desc_ = nullptr;
};

class DvppFrame {
 public:
  DvppFrame() = default;
  virtual ~DvppFrame() = default;

  std::shared_ptr<acldvppPicDesc> &GetPicDesc() { return pic_desc_; }

 private:
  std::shared_ptr<acldvppPicDesc> pic_desc_;
};

class DvppVideoDecodeContext {
 public:
  DvppVideoDecodeContext() {
    queue_ =
        std::make_shared<modelbox::BlockingQueue<std::shared_ptr<DvppFrame>>>();
  };

  virtual ~DvppVideoDecodeContext();

  std::shared_ptr<modelbox::BlockingQueue<std::shared_ptr<DvppFrame>>>
  GetCacheQueue() {
    return queue_;
  }

 private:
  std::shared_ptr<modelbox::BlockingQueue<std::shared_ptr<DvppFrame>>> queue_;
};

class AscendVideoDecoder {
 public:
  AscendVideoDecoder(int instance_id, int device_id, int32_t rate_num,
                     int32_t rate_den, int32_t format, int32_t entype);
  virtual ~AscendVideoDecoder();

  modelbox::Status Init(std::shared_ptr<modelbox::DataContext> data_ctx);

  modelbox::Status Decode(
      std::shared_ptr<DvppPacket> dvpp_packet,
      std::shared_ptr<DvppVideoDecodeContext> dvpp_decoder_ctx);

  int32_t GetRateNum() { return rate_num_; }
  int32_t GetRateDen() { return rate_den_; }

 private:
  static void Callback(acldvppStreamDesc *input, acldvppPicDesc *output,
                       void *userData);
  modelbox::Status ProcessLastPacket(
      std::shared_ptr<DvppPacket> dvpp_packet,
      std::shared_ptr<DvppVideoDecodeContext> dvpp_decoder_ctx);
  std::shared_ptr<acldvppPicDesc> SetUpFrame(
      std::shared_ptr<DvppPacket> dvpp_packet);
  int32_t instance_id_{0};
  int32_t device_id_{0};
  int32_t rate_num_{0};
  int32_t rate_den_{0};
  int32_t format_{1};
  int32_t entype_{2};
  std::shared_ptr<aclvdecChannelDesc> vdecChannelDesc_ = nullptr;
  std::shared_ptr<ThreadHandler> thread_handler_ = nullptr;
};

#endif