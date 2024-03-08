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

#ifndef MODELBOX_FLOWUNIT_VIDEO_DEMUXER_CPU_H_
#define MODELBOX_FLOWUNIT_VIDEO_DEMUXER_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include "ffmpeg_video_demuxer.h"
#include "modelbox/flowunit.h"
#include "source_context.h"

constexpr const char *FLOWUNIT_NAME = "video_demuxer";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: A video demuxer flowunit on cpu. \n"
    "\t@Port parameter: The input port buffer data indicate video file path or "
    "stream path, the output "
    "port buffer type is video_packet.\n"
    "\t  The video_packet buffer contain the following meta fields:\n"
    "\t\tField Name: pts,           Type: int64_t\n"
    "\t\tField Name: dts,           Type: int64_t\n"
    "\t\tField Name: rate_num,      Type: int32_t\n"
    "\t\tField Name: rate_den,      Type: int32_t\n"
    "\t\tField Name: duration,      Type: int64_t\n"
    "\t\tField Name: time_base,     Type: double\n"
    "\t\tField Name: width,         Type: int32_t\n"
    "\t\tField Name: height,        Type: int32_t\n"
    "\t@Constraint: The flowuint 'video_decoder' must be used pair "
    "with 'video_demuxer. ";
constexpr const char *SOURCE_URL = "source_url";
constexpr const char *CODEC_META = "codec_meta";
constexpr const char *PROFILE_META = "profile_meta";
constexpr const char *DEMUXER_CTX = "demuxer_ctx";
constexpr const char *STREAM_META_INPUT = "in_video_url";
constexpr const char *VIDEO_PACKET_OUTPUT = "out_video_packet";
constexpr const char *DEMUX_RETRY_CONTEXT = "source_context";
constexpr const char *DEMUX_TIMER_TASK = "demux_timer_task";

enum DemuxStatus { DEMUX_FAIL = 0, DEMUX_SUCCESS = 1 };

class VideoDemuxerFlowUnit
    : public modelbox::FlowUnit,
      public std::enable_shared_from_this<VideoDemuxerFlowUnit> {
 public:
  VideoDemuxerFlowUnit();
  ~VideoDemuxerFlowUnit() override;

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
  std::shared_ptr<std::string> GetSourceUrl(
      const std::shared_ptr<modelbox::DataContext> &data_ctx);

  modelbox::Status Reconnect(modelbox::Status &status,
                             std::shared_ptr<modelbox::DataContext> &data_ctx);
  modelbox::Status CreateRetryTask(
      std::shared_ptr<modelbox::DataContext> &data_ctx);
  modelbox::Status WriteData(
      std::shared_ptr<modelbox::DataContext> &data_ctx,
      std::shared_ptr<AVPacket> &pkt,
      const std::shared_ptr<FfmpegVideoDemuxer> &video_demuxer);
  void WriteEnd(std::shared_ptr<modelbox::DataContext> &data_ctx);

  modelbox::Status InitDemuxer(std::shared_ptr<modelbox::DataContext> &data_ctx,
                               std::shared_ptr<std::string> &source_url);

  void UpdateStatsInfo(const std::shared_ptr<modelbox::DataContext> &data_ctx,
                       const std::shared_ptr<FfmpegVideoDemuxer> &demuxer);

  bool key_frame_only_{false};
  size_t queue_size_{32};
  bool is_retry_reset_{false};
};

class DemuxerWorker {
 public:
  DemuxerWorker(bool is_async, size_t cache_size,
                std::shared_ptr<FfmpegVideoDemuxer> demuxer);

  virtual ~DemuxerWorker();

  modelbox::Status Init();

  std::shared_ptr<FfmpegVideoDemuxer> GetDemuxer() const;

  size_t GetDropCount() const;

  modelbox::Status ReadPacket(std::shared_ptr<AVPacket> &av_packet);

  bool IsRunning() const;

  void Process();

 private:
  void PushCache(const std::shared_ptr<AVPacket> &av_packet);

  modelbox::Status PopCache(std::shared_ptr<AVPacket> &av_packet);

  bool IsKeyFrame(const std::shared_ptr<AVPacket> &av_packet);

  bool is_async_{false};
  size_t cache_size_{0};
  std::shared_ptr<FfmpegVideoDemuxer> demuxer_;

  std::atomic_bool demux_thread_running_{false};
  std::shared_ptr<std::thread> demux_thread_;

  std::mutex packet_cache_lock_;
  std::condition_variable packet_cache_not_empty_;
  std::list<std::shared_ptr<AVPacket>> packet_cache_;
  modelbox::Status last_demux_status_;
  size_t packet_drop_count_{0};
  bool missing_pre_packet_{false};
};

#endif  // MODELBOX_FLOWUNIT_VIDEO_DEMUXER_CPU_H_
