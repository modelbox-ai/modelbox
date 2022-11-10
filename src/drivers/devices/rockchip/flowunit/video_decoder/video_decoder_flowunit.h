/*
 * Copyright 2022 The Modelbox Project Authors. All Rights Reserved.
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

#ifndef MODELBOX_FLOWUNIT_VIDEO_DECODER_ROCKCHIP_H_
#define MODELBOX_FLOWUNIT_VIDEO_DECODER_ROCKCHIP_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/base/timer.h>
#include <modelbox/buffer.h>
#include <modelbox/device/rockchip/device_rockchip.h>
#include <modelbox/device/rockchip/rockchip_api.h>
#include <modelbox/device/rockchip/rockchip_memory.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include <algorithm>
#include <atomic>
#include <thread>

#include "rga.h"

constexpr const char *FLOWUNIT_NAME = "video_decoder";
constexpr const char *FLOWUNIT_DESC = "A rockchip video decoder flowunit";
constexpr const char *VIDEO_PACKET_INPUT = "in_video_packet";
constexpr const char *FRAME_INFO_OUTPUT = "out_video_frame";
constexpr const char *CODEC_META = "codec_meta";
constexpr const char *DECODER_CTX = "decoder_ctx";
constexpr const char *FRAME_INDEX_CTX = "frame_index_ctx";

class VideoDecoderFlowUnit : public modelbox::FlowUnit {
 public:
  VideoDecoderFlowUnit() = default;
  ~VideoDecoderFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> ctx) override;
  modelbox::Status DataPre(std::shared_ptr<modelbox::DataContext> ctx) override;
  modelbox::Status DataPost(
      std::shared_ptr<modelbox::DataContext> ctx) override;
  modelbox::Status DataGroupPre(
      std::shared_ptr<modelbox::DataContext> ctx) override {
    return modelbox::STATUS_OK;
  }
  modelbox::Status DataGroupPost(
      std::shared_ptr<modelbox::DataContext> ctx) override {
    return modelbox::STATUS_OK;
  }

 private:
  std::shared_ptr<modelbox::Buffer> ColorChange(MppFrame &frame, int32_t ws,
                                                int32_t hs);
  modelbox::Status WriteData(const std::shared_ptr<modelbox::DataContext> &ctx,
                             std::shared_ptr<modelbox::Buffer> &pack_buff,
                             std::vector<MppFrame> &out_frame);

 private:
  size_t queue_size_{0};
  std::string out_pix_fmt_str_;
  RgaSURF_FORMAT out_pix_fmt_{RK_FORMAT_YCbCr_420_SP};
  std::mutex rk_dec_mtx_;
};

#endif  // MODELBOX_FLOWUNIT_VIDEO_DECODER_ROCKCHIP_H_
