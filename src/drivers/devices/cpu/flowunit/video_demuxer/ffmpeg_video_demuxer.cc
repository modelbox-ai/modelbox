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

#include "ffmpeg_video_demuxer.h"

#include <modelbox/base/log.h>

#include <algorithm>

#include "driver_util.h"

#define GET_FFMPEG_ERR(err_num, var_name)        \
  char var_name[AV_ERROR_MAX_STRING_SIZE] = {0}; \
  av_make_error_string(var_name, AV_ERROR_MAX_STRING_SIZE, err_num);

modelbox::Status FfmpegVideoDemuxer::Init(std::shared_ptr<FfmpegReader> &reader,
                                          bool key_frame_only) {
  source_url_ = reader->GetSourceURL();
  format_ctx_ = reader->GetCtx();

  if (format_ctx_ == nullptr) {
    return modelbox::STATUS_FAULT;
  }

  reader_ = reader;
  auto ret = SetupStreamInfo();
  if (ret != modelbox::STATUS_SUCCESS) {
    return ret;
  }

  ret = GetStreamParam();
  if (ret != modelbox::STATUS_SUCCESS) {
    return ret;
  }

  key_frame_only_ = key_frame_only;
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status FfmpegVideoDemuxer::Demux(
    std::shared_ptr<AVPacket> &av_packet) {
  if (format_ctx_ == nullptr) {
    MBLOG_ERROR << "ffmpeg format context is null, init first";
    return modelbox::STATUS_FAULT;
  }

  reader_->ResetStartTime();
  auto ret = ReadPacket(av_packet);
  if (ret != modelbox::STATUS_SUCCESS) {
    return ret;
  }

  ret = BsfProcess(av_packet);
  if (ret != modelbox::STATUS_SUCCESS) {
    return ret;
  }

  return modelbox::STATUS_SUCCESS;
}

void FfmpegVideoDemuxer::LogStreamInfo() {
  MBLOG_INFO << "demux info:";
  MBLOG_INFO << "source url: " << driverutil::string_masking(source_url_);
  MBLOG_INFO << "key frame only: " << key_frame_only_;
  MBLOG_INFO << "codec id: " << codec_id_;
  MBLOG_INFO << "profile id: " << profile_id_;
  MBLOG_INFO << "creation time: " << creation_time_;
  MBLOG_INFO << "time base: " << time_base_;
  MBLOG_INFO << "frame width: " << frame_width_;
  MBLOG_INFO << "frame height: " << frame_height_;
  MBLOG_INFO << "frame rate: " << frame_rate_numerator_ << "/"
             << frame_rate_denominator_;
  MBLOG_INFO << "frame rotate: " << frame_rotate_;
  MBLOG_INFO << "frame count: " << frame_count_;
  MBLOG_INFO << "video duration: " << GetDuration();
  std::stringstream bsf_name_log;
  for (auto &bsf_name : bsf_name_list_) {
    bsf_name_log << bsf_name << ",";
  }

  MBLOG_INFO << "bsf_name:" << bsf_name_log.str();
}

AVCodecID FfmpegVideoDemuxer::GetCodecID() { return codec_id_; }

int32_t FfmpegVideoDemuxer::GetProfileID() { return profile_id_; }

const AVCodecParameters *FfmpegVideoDemuxer::GetCodecParam() {
  return format_ctx_->streams[stream_id_]->codecpar;
}

void FfmpegVideoDemuxer::GetFrameRate(int32_t &rate_num, int32_t &rate_den) {
  rate_num = frame_rate_numerator_;
  rate_den = frame_rate_denominator_;
}

void FfmpegVideoDemuxer::GetFrameMeta(int32_t *frame_width,
                                      int32_t *frame_height) {
  *frame_width = frame_width_;
  *frame_height = frame_height_;
}

int32_t FfmpegVideoDemuxer::GetFrameRotate() { return frame_rotate_; }

double FfmpegVideoDemuxer::GetTimeBase() { return time_base_; }

int64_t FfmpegVideoDemuxer::GetDuration() {
  if (format_ctx_->duration != AV_NOPTS_VALUE) {
    return format_ctx_->duration / AV_TIME_BASE;
  }

  auto &stream = format_ctx_->streams[stream_id_];
  if (stream->duration > 0 && stream->time_base.den > 0) {
    return stream->duration / stream->time_base.den * stream->time_base.num;
  }

  return 0;
}

modelbox::Status FfmpegVideoDemuxer::ReadPacket(
    std::shared_ptr<AVPacket> &av_packet) {
  int32_t ret = 0;
  auto *packet_ptr = av_packet_alloc();
  if (packet_ptr == nullptr) {
    MBLOG_ERROR << "ReadPacket alloc packet failed";
    return modelbox::STATUS_FAULT;
  }

  av_packet.reset(packet_ptr,
                  [](AVPacket *packet) { av_packet_free(&packet); });
  while ((ret = av_read_frame(format_ctx_.get(), av_packet.get())) >= 0) {
    if (!IsTargetPacket(av_packet)) {
      av_packet_unref(av_packet.get());
      continue;
    }

    break;
  }

  if (ret == AVERROR_EOF) {
    MBLOG_INFO << "Stream " << driverutil::string_masking(source_url_) << " is end";
    return modelbox::STATUS_NODATA;
  }

  if (ret < 0) {
    GET_FFMPEG_ERR(ret, err_str);
    MBLOG_ERROR << "av_read_frame failed, err " << err_str;
    return modelbox::STATUS_FAULT;
  }

  if (av_packet->size < 0) {
    MBLOG_ERROR << "Read packet size < 0";
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_SUCCESS;
}

bool FfmpegVideoDemuxer::IsTargetPacket(std::shared_ptr<AVPacket> &av_packet) {
  if (av_packet->stream_index != stream_id_) {
    return false;
  }

  if (key_frame_only_ && ((av_packet->flags & AV_PKT_FLAG_KEY) != 0)) {
    return false;
  }

  if (av_packet->size == 0) {
    return false;
  }

  return true;
}

modelbox::Status FfmpegVideoDemuxer::BsfProcess(
    std::shared_ptr<AVPacket> &av_packet) {
  for (size_t i = 0; i < bsf_ctx_list_.size(); ++i) {
    auto &bsf_ctx = bsf_ctx_list_[i];
    auto &bsf_name = bsf_name_list_[i];
    if (bsf_ctx == nullptr) {
      continue;
    }

    auto ret = av_bsf_send_packet(bsf_ctx.get(), av_packet.get());
    if (ret < 0) {
      GET_FFMPEG_ERR(ret, err_str);
      MBLOG_ERROR << "Bit stream filter[" << bsf_name
                  << "] send packet failed, ret " << err_str;
      return modelbox::STATUS_FAULT;
    }

    ret = av_bsf_receive_packet(bsf_ctx.get(), av_packet.get());
    if (ret < 0) {
      GET_FFMPEG_ERR(ret, err_str);
      MBLOG_ERROR << "Bit stream filter[" << bsf_name
                  << "] receive packet failed, ret " << err_str;
      return modelbox::STATUS_FAULT;
    }
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status FfmpegVideoDemuxer::SetupStreamInfo() {
  auto ret = avformat_find_stream_info(format_ctx_.get(), nullptr);
  if (ret < 0) {
    GET_FFMPEG_ERR(ret, err_str);
    MBLOG_ERROR << "Find stream info failed, err " << err_str;
    return modelbox::STATUS_FAULT;
  }

  stream_id_ = av_find_best_stream(format_ctx_.get(), AVMEDIA_TYPE_VIDEO, -1,
                                   -1, nullptr, 0);
  if (stream_id_ < 0) {
    MBLOG_ERROR << "Count find a stream";
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status FfmpegVideoDemuxer::GetStreamParam() {
  auto ret = GetStreamCodecID();
  if (ret != modelbox::STATUS_SUCCESS) {
    return ret;
  }

  ret = GetStreamTimeInfo();
  if (ret != modelbox::STATUS_SUCCESS) {
    return ret;
  }

  ret = GetStreamFrameInfo();
  if (ret != modelbox::STATUS_SUCCESS) {
    return ret;
  }

  ret = GetStreamBsfInfo();
  if (ret != modelbox::STATUS_SUCCESS) {
    return ret;
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status FfmpegVideoDemuxer::GetStreamCodecID() {
  codec_id_ = format_ctx_->streams[stream_id_]->codecpar->codec_id;
  profile_id_ = format_ctx_->streams[stream_id_]->codecpar->profile & 0xFF;
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status FfmpegVideoDemuxer::GetStreamTimeInfo() {
  auto *entry =
      av_dict_get(format_ctx_->metadata, "creation_timestamp", nullptr, 0);
  if (entry != nullptr) {
    creation_time_ = atol(entry->value);
  } else {
    MBLOG_INFO << "Stream " << driverutil::string_masking(source_url_) << " creation time is null";
  }

  time_base_ = av_q2d(format_ctx_->streams[stream_id_]->time_base) * 1000;
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status FfmpegVideoDemuxer::GetStreamFrameInfo() {
  frame_width_ = format_ctx_->streams[stream_id_]->codecpar->width;
  frame_height_ = format_ctx_->streams[stream_id_]->codecpar->height;
  frame_rate_numerator_ = format_ctx_->streams[stream_id_]->avg_frame_rate.num;
  frame_rate_denominator_ =
      format_ctx_->streams[stream_id_]->avg_frame_rate.den;
  auto *entry = av_dict_get(format_ctx_->streams[stream_id_]->metadata,
                            "rotate", nullptr, 0);
  if (entry != nullptr) {
    frame_rotate_ = (atol(entry->value) % 360 + 360) % 360;
  } else {
    MBLOG_INFO << "Stream " << driverutil::string_masking(source_url_) << " rotate is null";
  }
  RescaleFrameRate(frame_rate_numerator_, frame_rate_denominator_);
  frame_count_ = format_ctx_->streams[stream_id_]->nb_frames;
  return modelbox::STATUS_SUCCESS;
}

void FfmpegVideoDemuxer::RescaleFrameRate(int32_t &frame_rate_numerator,
                                          int32_t &frame_rate_denominator) {
  // Try to avoid too large numerator & denominator
  const int32_t fraction_limit =
      32767;  // Try to be close to this value, might be greater
  auto numerator_scale = frame_rate_numerator / fraction_limit;
  auto denominator_scale = frame_rate_denominator / fraction_limit;
  auto fraction_scale = std::max(numerator_scale, denominator_scale);
  fraction_scale = std::min(fraction_scale, frame_rate_denominator);
  if (fraction_scale > 1) {
    frame_rate_numerator = frame_rate_numerator / fraction_scale;
    // We are ensured that fraction_scale <= frame_rate_denominator
    frame_rate_denominator = frame_rate_denominator / fraction_scale;
  }
}

modelbox::Status FfmpegVideoDemuxer::GetStreamBsfInfo() {
  auto *extra_data = format_ctx_->streams[stream_id_]->codecpar->extradata;
  auto extra_size = format_ctx_->streams[stream_id_]->codecpar->extradata_size;
  std::stringstream extra_data_log;
  for (int i = 0; i < extra_size; ++i) {
    extra_data_log << std::hex << int(extra_data[i]) << ":";
  }

  MBLOG_INFO << "extra_data: " << extra_data_log.str();
  std::string bsf_name;
  auto ret = GetBsfName(format_ctx_->streams[stream_id_]->codecpar->codec_tag,
                        codec_id_, extra_data, extra_size, bsf_name);
  if (ret != modelbox::STATUS_SUCCESS) {
    return modelbox::STATUS_FAULT;
  }

  bsf_name_list_.push_back(bsf_name);
  bsf_ctx_list_.push_back(CreateBsfCtx(bsf_name));
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status FfmpegVideoDemuxer::GetBsfName(
    uint32_t codec_tag, AVCodecID codec_id, uint8_t *extra_data,
    size_t extra_size, std::string &bsf_name) {
  char fourcc_str_array[AV_FOURCC_MAX_STRING_SIZE] = {0};
  char *fourcc = av_fourcc_make_string(fourcc_str_array, codec_tag);
  if (fourcc) {
    MBLOG_INFO << "try get bsf for Fourcc:" << fourcc
               << ", CodecId:" << codec_id;
  }
  // 1.Judge by codec_id
  if (codec_id == AV_CODEC_ID_H264) {
    bsf_name = "h264_mp4toannexb";
  } else if (codec_id == AV_CODEC_ID_H265) {
    bsf_name = "hevc_mp4toannexb";
  } else {
    // Try use dump_extra
    bsf_name = "dump_extra";
  }

  // 2.Judge by codec_tag & extra_data
  if (codec_tag == 0 && IsAnnexb(extra_data, extra_size)) {
    bsf_name = "dump_extra";
  }

  return modelbox::STATUS_SUCCESS;
}

bool FfmpegVideoDemuxer::IsAnnexb(const uint8_t *extra_data,
                                  size_t extra_size) {
  auto size_test = !extra_size;
  auto start_code1 = extra_size >= 3 && extra_data[0] == 0 &&
                     extra_data[1] == 0 && extra_data[2] == 1;
  auto start_code2 = extra_size >= 4 && extra_data[0] == 0 &&
                     extra_data[1] == 0 && extra_data[2] == 0 &&
                     extra_data[3] == 1;
  return size_test || start_code1 || start_code2;
}

std::shared_ptr<AVBSFContext> FfmpegVideoDemuxer::CreateBsfCtx(
    const std::string &bsf_name, AVDictionary **options) {
  const auto *bsf = av_bsf_get_by_name(bsf_name.c_str());
  if (!bsf) {
    MBLOG_ERROR << "Get bit stream filter failed, name " << bsf_name;
    return nullptr;
  }

  AVBSFContext *ctx = nullptr;
  auto ret = av_bsf_alloc(bsf, &ctx);
  if (ret < 0) {
    GET_FFMPEG_ERR(ret, err_str);
    MBLOG_ERROR << "Alloc bit stream filter context failed, name " << bsf_name
                << ", err " << err_str;
    return nullptr;
  }

  std::shared_ptr<AVBSFContext> bsf_ctx(
      ctx, [](AVBSFContext *ctx) { av_bsf_free(&ctx); });
  ret = avcodec_parameters_copy(bsf_ctx->par_in,
                                format_ctx_->streams[stream_id_]->codecpar);
  if (ret < 0) {
    GET_FFMPEG_ERR(ret, err_str);
    MBLOG_ERROR << "Copy codec param to bsf " << bsf_name << " failed, err "
                << err_str;
    return nullptr;
  }

  if (options) {
    ret = av_opt_set_dict2(bsf_ctx.get(), options, AV_OPT_SEARCH_CHILDREN);
    if (ret < 0) {
      GET_FFMPEG_ERR(ret, err_str);
      MBLOG_ERROR << "Set option to bsf " << bsf_name << " failed, err "
                  << err_str;
      return nullptr;
    }
  }

  ret = av_bsf_init(bsf_ctx.get());
  if (ret < 0) {
    GET_FFMPEG_ERR(ret, err_str);
    MBLOG_ERROR << "Init bsf " << bsf_name << " failed, err " << err_str;
    return nullptr;
  }

  return bsf_ctx;
}