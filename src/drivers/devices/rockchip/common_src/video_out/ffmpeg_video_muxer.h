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

#ifndef MODELBOX_FLOWUNIT_FFMPEG_MUXER_H_
#define MODELBOX_FLOWUNIT_FFMPEG_MUXER_H_

#include <modelbox/base/status.h>

#include <memory>
#include <string>

#include "ffmpeg_writer.h"

class FfmpegVideoMuxer {
 public:
  modelbox::Status Init(const std::shared_ptr<AVCodecContext> &codec_ctx,
                        const std::shared_ptr<FfmpegWriter> &writer);

  modelbox::Status Mux(const AVRational &time_base,
                       const std::shared_ptr<AVPacket> &av_packet);

  virtual ~FfmpegVideoMuxer();

 private:
  modelbox::Status SetupStreamParam(
      const std::shared_ptr<AVCodecContext> &codec_ctx);

  std::shared_ptr<AVFormatContext> format_ctx_;
  std::string destination_url_;
  AVStream *stream_{nullptr};
  bool is_header_wrote_{false};
};

#endif  // MODELBOX_FLOWUNIT_FFMPEG_MUXER_H_