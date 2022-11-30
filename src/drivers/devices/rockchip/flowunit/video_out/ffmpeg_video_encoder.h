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

#ifndef MODELBOX_FLOWUNIT_VIDEO_OUT_H_
#define MODELBOX_FLOWUNIT_VIDEO_OUT_H_

#include <modelbox/base/status.h>
#include <modelbox/device/rockchip/device_rockchip.h>
#include <modelbox/device/rockchip/rockchip_api.h>

#include <memory>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#ifdef __cplusplus
}
#endif

class FfmpegVideoEncoder {
 public:
  modelbox::Status Init(const std::shared_ptr<modelbox::Device> &device,
                        int32_t width, int32_t height,
                        const AVRational &frame_rate);

  modelbox::Status Encode(
      const std::shared_ptr<modelbox::Device> &device,
      const std::shared_ptr<AVFrame> &av_frame,
      std::vector<std::shared_ptr<AVPacket>> &av_packet_list);

  std::shared_ptr<AVCodecContext> GetCtx() { return av_codec_ctx_; }

 private:
  void SetupCodecParam(int32_t width, int32_t height,
                       const AVRational &frame_rate,
                       std::shared_ptr<AVCodecContext> &codec_ctx);

  std::shared_ptr<AVCodecContext> av_codec_ctx_;
  // rk
  modelbox::Status RkInit(int w, int h, const AVRational &frame_rate,
                          const std::string &encodeType);
  modelbox::Status Init_PrepConfig();
  modelbox::Status Init_RcConfig();
  modelbox::Status Init_CodecConfig();
  modelbox::Status Init_Config();
  modelbox::Status Init_MppContex();
  void CloseRkEncoder();
  std::shared_ptr<AVPacket> NewPacket(MppPacket &packet);
  std::shared_ptr<modelbox::Buffer> FromAvFrame(
      const std::shared_ptr<modelbox::Device> &device,
      const std::shared_ptr<AVFrame> &av_frame);

  MppCodingType codec_type_ = MPP_VIDEO_CodingAVC;
  MppCtx codec_ctx_ = nullptr;
  MppApi *rk_api_ = nullptr;
  MppEncCfg cfg_ = nullptr;

  int width_ = 0;
  int height_ = 0;
  int alignW_ = 0;
  int alignH_ = 0;
  int fps_ = 0;
  int fps_den_ = 0;
  int bps_ = 0;
  std::mutex rk_enc_mtx_;
};

#endif  // MODELBOX_FLOWUNIT_VIDEO_OUT_H_