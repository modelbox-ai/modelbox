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

#ifndef MODELBOX_FLOWUNIT_VIDEO_ENCODER_CPU_H_
#define MODELBOX_FLOWUNIT_VIDEO_ENCODER_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <vector>

#include "ffmpeg_video_encoder.h"
#include "ffmpeg_video_muxer.h"
#include "ffmpeg_writer.h"
#include "modelbox/flowunit.h"
#include "video_decode_common.h"

constexpr const char *FLOWUNIT_NAME = "video_encoder";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: A video encoder flowunit on cpu. \n"
    "\t@Port parameter: The input port buffer meta type is image \n"
    "\t  The image type buffer contains the following meta fields:\n"
    "\t\tField Name: width,         Type: int32_t\n"
    "\t\tField Name: height,        Type: int32_t\n"
    "\t\tField Name: width_stride,  Type: int32_t\n"
    "\t\tField Name: height_stride, Type: int32_t\n"
    "\t\tField Name: channel,       Type: int32_t\n"
    "\t\tField Name: pix_fmt,       Type: string\n"
    "\t\tField Name: layout,        Type: int32_t\n"
    "\t\tField Name: shape,         Type: vector<size_t>\n"
    "\t\tField Name: type,          Type: ModelBoxDataType::MODELBOX_UINT8\n"
    "\t@Constraint: The field value range of this flowunit supports: "
    "'pix_fmt': "
    "[rgb, bgr, nv12], 'layout': [hwc]. ";
constexpr const char *DEST_URL = "dest_url";
constexpr const char *COLOR_CVT_CTX = "color_cvt_ctx";
constexpr const char *FRAME_INDEX_CTX = "frame_index_ctx";
constexpr const char *ENCODER_CTX = "encoder_ctx";
constexpr const char *MUXER_CTX = "muxer_ctx";
constexpr const char *FORMAT_NAME = "format_name";
constexpr const char *CODEC_NAME = "codec_name";
constexpr const char *DESTINATION_URL = "destination_url";
constexpr const char *FRAME_INFO_INPUT = "in_video_frame";

class VideoEncoderFlowUnit : public modelbox::FlowUnit {
 public:
  VideoEncoderFlowUnit();
  ~VideoEncoderFlowUnit() override;

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
  modelbox::Status GetDestUrl(std::shared_ptr<modelbox::DataContext> &data_ctx,
                              std::string &dest_url);

  modelbox::Status ReadFrames(
      const std::shared_ptr<FfmpegColorConverter> &color_cvt,
      const std::shared_ptr<modelbox::DataContext> &data_ctx,
      std::vector<std::shared_ptr<AVFrame>> &av_frame_list);

  modelbox::Status ReadFrameFromBuffer(
      std::shared_ptr<modelbox::Buffer> &frame_buffer,
      std::shared_ptr<AVFrame> &av_frame);

  modelbox::Status CvtFrameToYUV420P(
      const std::shared_ptr<FfmpegColorConverter> &color_cvt,
      const std::shared_ptr<AVFrame> &origin,
      std::shared_ptr<AVFrame> &yuv420p_frame);

  modelbox::Status EncodeFrame(
      const std::shared_ptr<FfmpegVideoEncoder> &encoder,
      const std::vector<std::shared_ptr<AVFrame>> &av_frame_list,
      std::vector<std::shared_ptr<AVPacket>> &av_packet_list);

  modelbox::Status MuxPacket(
      const std::shared_ptr<FfmpegVideoMuxer> &muxer,
      const AVRational &time_base,
      std::vector<std::shared_ptr<AVPacket>> &av_packet_list);

  modelbox::Status OpenMuxer(
      const std::shared_ptr<modelbox::DataContext> &data_ctx, int32_t width,
      int32_t height, int32_t rate_num, int32_t rate_den, std::string dest_url);

  modelbox::Status CloseMuexer(
      const std::shared_ptr<modelbox::DataContext> &data_ctx);

  std::string default_dest_url_;
  std::string format_name_;
  std::string encoder_name_;
  uint64_t bit_rate_{0};
  bool reopen_remote_{false};
};

#endif  // MODELBOX_FLOWUNIT_VIDEO_ENCODER_CPU_H_
