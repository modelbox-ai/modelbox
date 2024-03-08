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

#ifndef MODELBOX_FLOWUNIT_DVPP_DECODE_ASCEND_H_
#define MODELBOX_FLOWUNIT_DVPP_DECODE_ASCEND_H_

#include <acl/ops/acl_dvpp.h>
#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/base/timer.h>
#include <modelbox/buffer.h>
#include <modelbox/device/ascend/device_ascend.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <algorithm>

#include "ascend_video_decode.h"

constexpr const char *FLOWUNIT_NAME = "video_decoder";
constexpr const char *FLOWUNIT_TYPE = "ascend";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: A resize flowunit on cpu. \n"
    "\t@Port parameter: the input port buffer type is video_packet, the output "
    "port buffer type is video_frame.\n"
    "\t  The video_packet buffer contain the following meta fields:\n"
    "\t\tField Name: pts,           Type: int64_t\n"
    "\t\tField Name: dts,           Type: int64_t\n"
    "\t\tField Name: rate_num,      Type: int32_t\n"
    "\t\tField Name: rate_den,      Type: int32_t\n"
    "\t\tField Name: duration,      Type: int64_t\n"
    "\t\tField Name: time_base,     Type: double\n"
    "\t\tField Name: width,         Type: int32_t\n"
    "\t\tField Name: height,        Type: int32_t\n"
    "\t  The video_frame buffer contain the following meta fields:\n"
    "\t\tField Name: index,         Type: int64_t\n"
    "\t\tField Name: rate_num,      Type: int32_t\n"
    "\t\tField Name: rate_den,      Type: int32_t\n"
    "\t\tField Name: duration,      Type: int64_t\n"
    "\t\tField Name: url,           Type: string\n"
    "\t\tField Name: timestamp,     Type: int64_t\n"
    "\t\tField Name: eos,           Type: bool\n"
    "\t\tField Name: width,         Type: int32_t\n"
    "\t\tField Name: height,        Type: int32_t\n"
    "\t\tField Name: width_stride,  Type: int32_t\n"
    "\t\tField Name: height_stride, Type: int32_t\n"
    "\t\tField Name: channel,       Type: int32_t\n"
    "\t\tField Name: pix_fmt,       Type: string\n"
    "\t\tField Name: layout,        Type: int32_t\n"
    "\t\tField Name: shape,         Type: vector<size_t>\n"
    "\t\tField Name: type,          Type: ModelBoxDataType::MODELBOX_UINT8\n"
    "\t@Constraint: the flowuint 'video_decoder' must be used pair "
    "with 'video_demuxer. the output buffer meta fields 'pix_fmt' is 'nv12', "
    "'layout' is 'hcw'.";
constexpr const char *DVPP_DECODER = "dvpp_decode";
constexpr const char *VIDEO_PACKET_INPUT = "in_video_packet";
constexpr const char *FRAME_INFO_OUTPUT = "out_video_frame";
constexpr const char *CODEC_META = "codec_meta";
constexpr const char *SOURCE_URL_META = "source_url";
constexpr const char *CODEC_ID_META = "codec_id";
constexpr const char *PROFILE_META = "profile_meta";
constexpr const char *DVPP_DECODER_CTX = "dvpp_decode_context";
constexpr const char *DVPP_DECODE_FLOWUNIT_DESC =
    "A dvpp_decode flowunit on Ascend";
constexpr const char *DVPP_DECODE_FLOWUNIT_NAME = "video_decoder";
constexpr const char *FRAME_INDEX_CTX = "frame_index_ctx";
constexpr const int DECODER_RETRY_NUM = 3;

// 此处应该是dvpp类型，但是当前不支持，先使用ascend
constexpr const char *DVPP_FLOWUNIT_TYPE = "ascend" /*"ascend-devpp"*/;
constexpr const char *DEVICE_DVPP_TYPE = "ascend" /*"ascend-devpp"*/;

class VideoDecodeFlowUnit : public modelbox::FlowUnit {
 public:
  VideoDecodeFlowUnit();
  ~VideoDecodeFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override;

  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status DataPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status DataPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  modelbox::Status DataGroupPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override {
    return modelbox::STATUS_OK;
  };

  modelbox::Status DataGroupPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override {
    return modelbox::STATUS_OK;
  };

 private:
  int32_t GetDvppEncodeType(AVCodecID codec_id, int32_t profile_id);
  modelbox::Status GetDecoderParam(
      const std::shared_ptr<modelbox::DataContext> &data_ctx, int32_t &rate_num,
      int32_t &rate_den, int32_t &encode_type);
  modelbox::Status ReadData(
      const std::shared_ptr<modelbox::DataContext> &data_ctx,
      std::vector<std::shared_ptr<DvppPacket>> &dvpp_packet_list,
      std::shared_ptr<modelbox::Buffer> &flag_buffer);
  modelbox::Status ReadDvppStreamDesc(
      const std::shared_ptr<modelbox::Buffer> &packet_buffer,
      std::shared_ptr<DvppPacket> &dvpp_packet);
  modelbox::Status SetUpTheLastPacket(std::shared_ptr<DvppPacket> &dvpp_packet);
  modelbox::Status WriteData(
      const std::shared_ptr<modelbox::DataContext> &data_ctx,
      const std::shared_ptr<AscendVideoDecoder> &video_decoder,
      const std::shared_ptr<DvppVideoDecodeContext> &dvpp_ctx);
  void InitInstanceId();
  int32_t FindTheMinimumAvailableId();
  void RestoreInstanceId(int32_t instance_id);

  modelbox::Status CloseDecoder(
      std::shared_ptr<modelbox::DataContext> &data_ctx);
  modelbox::Status NewDecoder(std::shared_ptr<modelbox::DataContext> &data_ctx,
                              const std::string &source_url, AVCodecID codec_id,
                              int32_t rate_num, int32_t rate_den,
                              int32_t encode_type);
  modelbox::Status ReopenDecoder(
      std::shared_ptr<modelbox::DataContext> &data_ctx,
      const std::shared_ptr<modelbox::Buffer> &flag_buffer);

  uint32_t dest_width_{224};
  uint32_t dest_height_{224};
  // 1: YUV420 semi-planner（nv12), 2: YVU420 semi-planner（nv21)
  int32_t format_{0};
  acldvppChannelDesc *dvpp_channel_desc_{nullptr};
  std::mutex mutex;
  std::map<int32_t, bool> instance_available_map_;
};

#endif  // MODELBOX_FLOWUNIT_DVPP_DECODE_ASCEND_H_
