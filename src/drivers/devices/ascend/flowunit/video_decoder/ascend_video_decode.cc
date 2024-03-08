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

#include "ascend_video_decode.h"

#include <fstream>
#include <string>

#include "modelbox/device/ascend/device_ascend.h"
#define ACL_ENABLE
#include "image_process.h"

void DestroyStreamDesc(acldvppStreamDesc *stream_desc) {
  if (stream_desc == nullptr) {
    return;
  }

  auto *data_ptr = acldvppGetStreamDescData(stream_desc);
  if (data_ptr != nullptr) {
    acldvppFree(data_ptr);
  }

  auto ret = acldvppDestroyStreamDesc(stream_desc);
  if (ret != ACL_ERROR_NONE) {
    MBLOG_ERROR << "fail to destroy stream desc, err code " << ret;
  }
}

void DestroyPicDesc(acldvppPicDesc *pic_desc) {
  if (pic_desc == nullptr) {
    return;
  }

  auto *data_ptr = acldvppGetPicDescData(pic_desc);
  if (data_ptr != nullptr) {
    acldvppFree(data_ptr);
  }

  auto ret = acldvppDestroyPicDesc(pic_desc);
  if (ret != ACL_ERROR_NONE) {
    MBLOG_ERROR << "destroy pic desc failed, err code " << ret;
  }
}

ThreadHandler::ThreadHandler(
    int device_id, int instance_id,
    const std::shared_ptr<modelbox::DataContext> &data_ctx)
    : device_id_(device_id), instance_id_(instance_id), data_ctx_(data_ctx) {}

ThreadHandler::~ThreadHandler() {
  UpdateNeedStop();
  pthread_join(threadId_, nullptr);
}

void *ThreadHandler::ThreadFunc(void *arg) {
  auto *thread_handler = (ThreadHandler *)arg;
  auto ret = aclrtSetDevice(thread_handler->device_id_);
  if (ret != ACL_ERROR_NONE) {
    MBLOG_ERROR << "acl set device " << thread_handler->device_id_ << " failed";
    return ((void *)(-1));
  }

  bool pstart = false;
  while (!thread_handler->need_stop_) {
    aclError ret = aclrtProcessReport(ACL_PROCESS_WAIT_TIME_OUT_MS);
    if (ret == ACL_ERROR_THREAD_NOT_SUBSCRIBE ||
        (ret == ACL_ERROR_WAIT_CALLBACK_TIMEOUT && !pstart)) {
      continue;
    }

    if (ret == ACL_ERROR_NONE) {
      pstart = true;
      continue;
    }

    pstart = false;
  }

  return nullptr;
}

modelbox::Status ThreadHandler::CreateThread() {
  int createThreadErr = pthread_create(&threadId_, nullptr,
                                       ThreadHandler::ThreadFunc, (void *)this);
  if (createThreadErr != 0) {
    const auto *errMsg = "create thread failed, err = " + createThreadErr;
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  return modelbox::STATUS_SUCCESS;
}

void ThreadHandler::UpdateNeedStop() { need_stop_ = true; }

DvppVideoDecodeContext::~DvppVideoDecodeContext() {
  std::vector<std::shared_ptr<DvppFrame>> not_consumed_frame_list;
  queue_->PopBatch(&not_consumed_frame_list, -1);
  for (auto &frame : not_consumed_frame_list) {
    auto *pic_desc = frame->GetPicDesc().get();
    if (pic_desc == nullptr) {
      continue;
    }

    auto *data = acldvppGetPicDescData(pic_desc);
    if (data == nullptr) {
      continue;
    }

    auto ret = acldvppFree(data);
    if (ret != ACL_SUCCESS) {
      MBLOG_WARN << "acl dvpp free failed, addr " << (void *)data << ", ret "
                 << ret;
    }
  }
}

AscendVideoDecoder::AscendVideoDecoder(int instance_id, int device_id,
                                       int32_t rate_num, int32_t rate_den,
                                       int32_t format, int32_t entype)
    : instance_id_(instance_id),
      device_id_(device_id),
      rate_num_(rate_num),
      rate_den_(rate_den),
      format_(format),
      entype_(entype){};

AscendVideoDecoder::~AscendVideoDecoder() {
  vdecChannelDesc_ = nullptr;
  thread_handler_ = nullptr;
};

void AscendVideoDecoder::Callback(acldvppStreamDesc *input,
                                  acldvppPicDesc *output, void *userData) {
  if (input == nullptr) {
    MBLOG_WARN << "dvpp decoder callback input is nullptr";
    return;
  }

  void *vdecInBufferDev = acldvppGetStreamDescData(input);
  if (vdecInBufferDev != nullptr) {
    aclError ret = acldvppFree(vdecInBufferDev);
    if (ret != ACL_ERROR_NONE) {
      MBLOG_ERROR << "fail to free input stream desc data, err code " << ret;
    }
  }

  aclError des_ret = acldvppDestroyStreamDesc(input);
  if (des_ret != ACL_ERROR_NONE) {
    MBLOG_ERROR << "fail to destroy input stream desc, err code " << des_ret;
  }

  if (output == nullptr) {
    MBLOG_WARN << "dvpp decoder callback output is nullptr.";
    return;
  }

  auto dvpp_frame = std::make_shared<DvppFrame>();
  dvpp_frame->GetPicDesc().reset(output, [](acldvppPicDesc *picDesc) {
    // will not free pic buffer
    auto ret = acldvppDestroyPicDesc(picDesc);
    if (ret != ACL_ERROR_NONE) {
      MBLOG_ERROR << "destroy pic desc failed, err code " << ret;
    }
  });

  void *vdecOutBufferDev = acldvppGetPicDescData(output);
  if (vdecOutBufferDev == nullptr) {
    MBLOG_ERROR << "dvpp decoder callback output data is nullptr.";
    return;
  }

  auto output_size = acldvppGetPicDescSize(output);
  if (output_size == 0) {
    acldvppFree(vdecOutBufferDev);
    MBLOG_ERROR << "dvpp decoder callback output size is zero.";
    return;
  }

  auto acl_ret = acldvppGetPicDescRetCode(output);
  if (acl_ret != 0) {
    acldvppFree(vdecOutBufferDev);
    MBLOG_ERROR << "vdec failed, err code " << acl_ret;
    return;
  }

  if (userData == nullptr) {
    acldvppFree(vdecOutBufferDev);
    MBLOG_ERROR << "call back userData is nullptr.";
    return;
  }

  auto *ctx = (DvppVideoDecodeContext *)userData;
  auto queue = ctx->GetCacheQueue();
  if (queue == nullptr) {
    acldvppFree(vdecOutBufferDev);
    MBLOG_ERROR << "get cache queue failed.";
    return;
  }
  auto res = queue->Push(dvpp_frame);
  if (!res) {
    acldvppFree(vdecOutBufferDev);
    MBLOG_INFO << "dvpp video decoder callback queue push failed.";
  }
}

modelbox::Status AscendVideoDecoder::Init(
    const std::shared_ptr<modelbox::DataContext> &data_ctx) {
  vdecChannelDesc_ = nullptr;
  aclError ret = aclrtSetDevice(device_id_);
  if (ret != ACL_ERROR_NONE) {
    auto errMsg = "acl set device " + std::to_string(device_id_) +
                  " failed, err code" + std::to_string(ret);
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  thread_handler_ =
      std::make_shared<ThreadHandler>(device_id_, instance_id_, data_ctx);
  auto status = thread_handler_->CreateThread();
  if (status != modelbox::STATUS_SUCCESS) {
    auto errMsg = "create thread failed, " + status.WrapErrormsgs();
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  bool setup_result = false;
  DeferCond { return !setup_result; };

  DeferCondAdd { thread_handler_ = nullptr; };

  aclvdecChannelDesc *vdecChannelDescPtr = aclvdecCreateChannelDesc();
  if (vdecChannelDescPtr == nullptr) {
    const auto *errMsg =
        "fail to create vdec channel desc, pls check npu log for more details.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  DeferCondAdd {
    ret = aclvdecDestroyChannelDesc(vdecChannelDescPtr);
    if (ret != ACL_ERROR_NONE) {
      auto errMsg =
          "fail to destroy channel desc, err code " + std::to_string(ret);
      MBLOG_ERROR << errMsg;
    }
  };

  ret = aclvdecSetChannelDescChannelId(vdecChannelDescPtr, instance_id_);
  if (ret != ACL_ERROR_NONE) {
    auto errMsg = "fail to set vdec ChannelId, err code " + std::to_string(ret);
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  ret = aclvdecSetChannelDescThreadId(vdecChannelDescPtr,
                                      thread_handler_->GetThreadId());
  if (ret != ACL_ERROR_NONE) {
    auto errMsg = "fail to create threadId, err code " + std::to_string(ret);
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  // callback func
  ret = aclvdecSetChannelDescCallback(vdecChannelDescPtr, Callback);
  if (ret != ACL_ERROR_NONE) {
    auto errMsg = "fail to set vdec Callback, err code " + std::to_string(ret);
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  ret = aclvdecSetChannelDescEnType(vdecChannelDescPtr,
                                    static_cast<acldvppStreamFormat>(entype_));
  if (ret != ACL_ERROR_NONE) {
    auto errMsg = "fail to set vdec EnType, err code " + std::to_string(ret);
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  ret = aclvdecSetChannelDescOutPicFormat(
      vdecChannelDescPtr, static_cast<acldvppPixelFormat>(format_));
  if (ret != ACL_ERROR_NONE) {
    auto errMsg =
        "fail to set vdec OutPicFormat, err code " + std::to_string(ret);
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  // create vdec channel
  ret = aclvdecCreateChannel(vdecChannelDescPtr);
  if (ret != ACL_ERROR_NONE) {
    auto errMsg =
        "fail to create vdec channel, err code " + std::to_string(ret);
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  auto device_id = device_id_;
  vdecChannelDesc_.reset(
      vdecChannelDescPtr, [this, device_id](aclvdecChannelDesc *p) {
        auto ret = aclvdecDestroyChannel(p);
        if (ret != ACL_ERROR_NONE) {
          MBLOG_ERROR << "fail to destroy vdec channel, err: " << ret;
        }
        ret = aclvdecDestroyChannelDesc(p);
        if (ret != ACL_ERROR_NONE) {
          MBLOG_ERROR << "fail to destroy vdec channel desc, err: " << ret;
        }

        this->thread_handler_ = nullptr;
      });

  setup_result = true;

  return modelbox::STATUS_SUCCESS;
}

std::shared_ptr<acldvppPicDesc> AscendVideoDecoder::SetUpFrame(
    const std::shared_ptr<DvppPacket> &dvpp_packet) {
  auto width = dvpp_packet->GetWidth();
  auto height = dvpp_packet->GetHeight();
  auto align_width =
      imageprocess::align_up(width, imageprocess::ASCEND_WIDTH_ALIGN);
  auto align_height =
      imageprocess::align_up(height, imageprocess::ASCEND_HEIGHT_ALIGN);
  auto width_stride = 0;
  auto ret =
      imageprocess::GetWidthStride(OUTPUT_PIX_FMT, align_width, width_stride);
  if (!ret) {
    MBLOG_ERROR << "Get width stride failed, ret " << ret;
    return nullptr;
  }

  size_t size = 0;
  ret = imageprocess::GetImageBytes(OUTPUT_PIX_FMT, align_width, align_height,
                                    size);
  if (!ret) {
    MBLOG_ERROR << "Get image bytes failed, ret " << ret;
    return nullptr;
  }

  auto dvpp_pic_desc_ptr = CreateImgDesc(
      size, OUTPUT_PIX_FMT,
      imageprocess::ImageShape{width, height, width_stride, align_height},
      imageprocess::ImgDescDestroyFlag::NONE);
  if (dvpp_pic_desc_ptr == nullptr) {
    MBLOG_ERROR << "Create image desc failed, ret " << modelbox::StatusError;
    return nullptr;
  }

  return dvpp_pic_desc_ptr;
}

modelbox::Status AscendVideoDecoder::ProcessLastPacket(
    const std::shared_ptr<DvppPacket> &dvpp_packet,
    const std::shared_ptr<DvppVideoDecodeContext> &dvpp_decoder_ctx) {
  MBLOG_INFO << "process the last packet.";

  aclError ret = ACL_ERROR_NONE;

  ret = aclvdecSendFrame(vdecChannelDesc_.get(), dvpp_packet->GetStreamDesc(),
                         nullptr, nullptr, (void *)dvpp_decoder_ctx.get());
  if (ret != ACL_ERROR_NONE) {
    MBLOG_ERROR << "send eos frame failed, err code: " << ret;
    DestroyStreamDesc(dvpp_packet->GetStreamDesc());
  }

  return modelbox::STATUS_NODATA;
}

modelbox::Status AscendVideoDecoder::Decode(
    const std::shared_ptr<DvppPacket> &dvpp_packet,
    const std::shared_ptr<DvppVideoDecodeContext> &dvpp_decoder_ctx) {
  aclError ret = aclrtSetDevice(device_id_);
  if (ret != ACL_ERROR_NONE) {
    auto errMsg = "acl set device " + std::to_string(device_id_) + " failed";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  if (dvpp_packet->IsEnd()) {
    auto status = ProcessLastPacket(dvpp_packet, dvpp_decoder_ctx);
    if (status == modelbox::STATUS_FAULT) {
      return {modelbox::STATUS_FAULT, "send the last packet failed."};
    }
    return status;
  }

  auto pic_desc = SetUpFrame(dvpp_packet);
  if (pic_desc == nullptr) {
    const auto *errMsg = "set up frame failed";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  ret =
      aclvdecSendFrame(vdecChannelDesc_.get(), dvpp_packet->GetStreamDesc(),
                       pic_desc.get(), nullptr, (void *)dvpp_decoder_ctx.get());

  if (ret != ACL_ERROR_NONE) {
    MBLOG_ERROR << "send vdec frame failed, err code " << ret;
    DestroyStreamDesc(dvpp_packet->GetStreamDesc());
    DestroyPicDesc(pic_desc.get());
  }

  return modelbox::STATUS_SUCCESS;
}