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


#ifndef MODELBOX_FLOWUNIT_FFMPEG_VIDEO_DEMUXER_H_
#define MODELBOX_FLOWUNIT_FFMPEG_VIDEO_DEMUXER_H_

#include <modelbox/base/status.h>

#include <functional>
#include <memory>
#include <vector>

#include "ffmpeg_reader.h"
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/opt.h>
}

class FfmpegVideoDemuxer {
 public:
  modelbox::Status Init(std::shared_ptr<FfmpegReader> &reader,
                      bool key_frame_only);

  modelbox::Status Demux(std::shared_ptr<AVPacket> &av_packet);

  void LogStreamInfo();

  AVCodecID GetCodecID();

  int GetProfileID();

  const AVCodecParameters *GetCodecParam();

  void GetFrameRate(int32_t &rate_num, int32_t &rate_den);

  void GetFrameMeta(int32_t *frame_width, int32_t *frame_height);

  double GetTimeBase();

  int64_t GetDuration();

 private:
  void PrintCurrentOption(AVDictionary *options);

  modelbox::Status SetupStreamInfo();

  modelbox::Status GetStreamParam();

  modelbox::Status GetStreamCodecID();

  modelbox::Status GetStreamTimeInfo();

  modelbox::Status GetStreamFrameInfo();

  void RescaleFrameRate(int32_t &numerator_scale, int32_t &denominator_scale);

  modelbox::Status GetStreamBsfInfo();

  modelbox::Status ReadPacket(std::shared_ptr<AVPacket> &av_packet);

  bool IsTargetPacket(std::shared_ptr<AVPacket> &av_packet);

  modelbox::Status BsfProcess(std::shared_ptr<AVPacket> &av_packet);

  modelbox::Status GetBsfName(uint32_t codec_tag, AVCodecID codec_id,
                            uint8_t *extra_data, size_t extra_size,
                            std::string &bsf_name);

  std::shared_ptr<AVBSFContext> CreateBsfCtx(const std::string &bsf_name,
                                             AVDictionary **options = nullptr);

  bool IsAnnexb(uint8_t *extra_data, size_t extra_size);

  std::string source_url_;
  bool key_frame_only_{false};
  std::shared_ptr<AVFormatContext> format_ctx_;
  int32_t stream_id_{0};
  AVCodecID codec_id_{AVCodecID::AV_CODEC_ID_H264};
  int32_t profile_id_{0};
  int64_t creation_time_{0};
  double time_base_{0};
  int32_t frame_width_{0};
  int32_t frame_height_{0};
  int32_t frame_rate_numerator_{0};
  int32_t frame_rate_denominator_{0};
  int32_t frame_count_{0};
  std::vector<std::shared_ptr<AVBSFContext>> bsf_ctx_list_;
  std::vector<std::string> bsf_name_list_;
  std::shared_ptr<FfmpegReader> reader_;
};

#endif  // MODELBOX_FLOWUNIT_FFMPEG_VIDEO_DEMUXER_H_