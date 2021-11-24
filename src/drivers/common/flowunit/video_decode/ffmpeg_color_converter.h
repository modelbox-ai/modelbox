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


#ifndef MODELBOX_FLOWUNIT_FFMPEG_COLOR_CONVERTER_H_
#define MODELBOX_FLOWUNIT_FFMPEG_COLOR_CONVERTER_H_

#include <modelbox/base/status.h>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

class FfmpegColorConverter {
 public:
  modelbox::Status CvtColor(const std::shared_ptr<AVFrame> &src_frame,
                          uint8_t *out_frame_data, AVPixelFormat out_pix_fmt);

 private:
  bool SupportCvtPixFmt(AVPixelFormat pix_fmt);

  modelbox::Status InitSwsCtx(int32_t width, int32_t height,
                            AVPixelFormat src_pix_fmt, AVPixelFormat dest_pix_fmt);

  modelbox::Status GetLineSize(AVPixelFormat pix_fmt, int32_t width,
                             int32_t linesize[4], int32_t linesize_size);

  modelbox::Status AllocFrame(std::shared_ptr<AVFrame> &frame, int32_t *line_size,
                            int32_t width, int32_t height,
                            AVPixelFormat pix_fmt);

  std::shared_ptr<SwsContext> sws_ctx_;
  int32_t width_{0};
  int32_t height_{0};
};

#endif  // MODELBOX_FLOWUNIT_FFMPEG_COLOR_CONVERTER_H_