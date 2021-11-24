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


#ifndef MODELBOX_FLOWUNIT_FFMPEG_DECODER_H_
#define MODELBOX_FLOWUNIT_FFMPEG_DECODER_H_

#include <modelbox/base/status.h>
#include <memory>
#include <vector>
#include <list>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
}

class FfmpegVideoDecoder {
 public:
  modelbox::Status Init(AVCodecID codec_id);

  modelbox::Status Decode(const std::shared_ptr<const AVPacket> &av_packet,
                        std::list<std::shared_ptr<AVFrame>> &av_frame_list);

 private:
  AVCodecID codec_id_{AV_CODEC_ID_NONE};
  std::shared_ptr<AVCodecContext> av_ctx_;
};

#endif  // MODELBOX_FLOWUNIT_FFMPEG_DECODER_H_