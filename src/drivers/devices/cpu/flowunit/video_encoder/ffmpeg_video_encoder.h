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

#ifndef MODELBOX_FLOWUNIT_FFMPEG_ENCODER_H_
#define MODELBOX_FLOWUNIT_FFMPEG_ENCODER_H_

#include <modelbox/base/status.h>

#include <memory>
#include <vector>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

class FfmpegVideoEncoder {
 public:
  modelbox::Status Init(int32_t width, int32_t height,
                        const AVRational &frame_rate, uint64_t bit_rate,
                        const std::string &encoder_name);

  modelbox::Status Encode(
      const std::shared_ptr<AVFrame> &av_frame,
      std::vector<std::shared_ptr<AVPacket>> &av_packet_list);

  std::shared_ptr<AVCodecContext> GetCtx() { return codec_ctx_; }

 private:
  void SetupCodecParam(int32_t width, int32_t height,
                       const AVRational &frame_rate, uint64_t bit_rate,
                       AVDictionary *&param,
                       std::shared_ptr<AVCodecContext> &codec_ctx);

  std::shared_ptr<AVCodecContext> codec_ctx_;
};

#endif  // MODELBOX_FLOWUNIT_FFMPEG_ENCODER_H_