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

#ifndef MODELBOX_FLOWUNIT_VIDEO_DECODER_CPU_H_
#define MODELBOX_FLOWUNIT_VIDEO_DECODER_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include "modelbox/flowunit.h"
#include "nvcodec_video_decoder.h"

constexpr const char *FLOWUNIT_NAME = "video_decoder";
constexpr const char *FLOWUNIT_TYPE = "cuda";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: A video decoder flowunit on cpu. \n"
    "\t@Port parameter: The input port buffer type is video_packet, the output "
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
    "\t@Constraint: The flowuint 'video_decoder' must be used pair "
    "with 'video_demuxer. the output buffer meta fields 'pix_fmt' is "
    "'brg_packed' or 'rgb_packed', 'layout' is 'hcw'.";
constexpr const char *CODEC_META = "codec_meta";
constexpr const char *DECODER_CTX = "decoder_ctx";
constexpr const char *CVT_CTX = "converter_ctx";
constexpr const char *FRAME_INDEX_CTX = "frame_index_ctx";
constexpr const char *VIDEO_PACKET_INPUT = "in_video_packet";
constexpr const char *FRAME_INFO_OUTPUT = "out_video_frame";
constexpr const char *SOURCE_URL_META = "source_url";
constexpr const char *CODEC_ID_META = "codec_id";
constexpr const char *LAST_FRAME = "last_frame";

class VideoDecoderFlowUnit : public modelbox::FlowUnit {
 public:
  VideoDecoderFlowUnit();
  ~VideoDecoderFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override;
  /* run when processing data */
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
  modelbox::Status ReadData(
      const std::shared_ptr<modelbox::DataContext> &data_ctx,
      std::vector<std::shared_ptr<NvcodecPacket>> &pkt_list,
      std::shared_ptr<modelbox::Buffer> &flag_buffer);
  modelbox::Status ReadNvcodecPacket(
      const std::shared_ptr<modelbox::Buffer> &packet_buffer,
      std::shared_ptr<NvcodecPacket> &pkt);
  modelbox::Status WriteData(
      std::shared_ptr<modelbox::DataContext> &data_ctx,
      std::vector<std::shared_ptr<NvcodecFrame>> &frame_list, bool eos,
      const std::string &file_url);
  modelbox::Status CreateCudaContext(CUcontext &cu_ctx, std::string &device_id);

  modelbox::Status CloseDecoder(
      std::shared_ptr<modelbox::DataContext> &data_ctx);
  modelbox::Status NewDecoder(std::shared_ptr<modelbox::DataContext> &data_ctx,
                              const std::string &source_url,
                              AVCodecID codec_id);
  modelbox::Status ReopenDecoder(
      std::shared_ptr<modelbox::DataContext> &data_ctx,
      const std::shared_ptr<modelbox::Buffer> &flag_buffer);

  std::string out_pix_fmt_str_;
  bool skip_err_frame_{false};
  std::string device_id_;
  // limit decode concurrency to avoid decoder stuck bug in gpu driver
  uint32_t concurrency_limit_{0};
};

#endif  // MODELBOX_FLOWUNIT_VIDEO_DECODER_CPU_H_
