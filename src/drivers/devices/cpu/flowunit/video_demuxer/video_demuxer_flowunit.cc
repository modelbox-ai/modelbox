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

#include "video_demuxer_flowunit.h"

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

VideoDemuxerFlowUnit::VideoDemuxerFlowUnit() = default;
VideoDemuxerFlowUnit::~VideoDemuxerFlowUnit() = default;

modelbox::Status VideoDemuxerFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  key_frame_only_ = opts->GetBool("key_frame_only", false);
  queue_size_ = opts->GetUint64("queue_size", queue_size_);
  return modelbox::STATUS_OK;
}
modelbox::Status VideoDemuxerFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status VideoDemuxerFlowUnit::Reconnect(
    modelbox::Status &status,
    std::shared_ptr<modelbox::DataContext> &data_ctx) {
  auto ret = modelbox::STATUS_CONTINUE;
  DeferCond { return ret == modelbox::STATUS_SUCCESS; };
  DeferCondAdd { WriteEnd(data_ctx); };
  auto source_context = std::static_pointer_cast<modelbox::SourceContext>(
      data_ctx->GetPrivate(DEMUX_RETRY_CONTEXT));
  if (source_context == nullptr) {
    if (status == modelbox::STATUS_NODATA) {
      ret = modelbox::STATUS_SUCCESS;
      return ret;
    }
    return status;
  }

  source_context->SetLastProcessStatus(status);
  auto retry_status = source_context->NeedRetry();
  if (retry_status == modelbox::RETRY_NONEED) {
    ret = modelbox::STATUS_FAULT;
  } else if (retry_status == modelbox::RETRY_STOP) {
    ret = modelbox::STATUS_SUCCESS;
  } else {
    auto timer_task = std::static_pointer_cast<modelbox::TimerTask>(
        data_ctx->GetPrivate(DEMUX_TIMER_TASK));
    modelbox::TimerGlobal::Schedule(timer_task,
                                    source_context->GetRetryInterval(), 0);
  }
  return ret;
}

modelbox::Status VideoDemuxerFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto demuxer_worker = std::static_pointer_cast<DemuxerWorker>(
      data_ctx->GetPrivate(DEMUXER_CTX));
  modelbox::Status demux_status = modelbox::STATUS_FAULT;
  std::shared_ptr<AVPacket> pkt;
  if (demuxer_worker != nullptr) {
    demux_status = demuxer_worker->ReadPacket(pkt);
    if (demux_status == modelbox::STATUS_NODATA) {
      is_retry_reset_ = true;
    }
  }

  if (demux_status == modelbox::STATUS_OK) {
    auto video_demuxer = demuxer_worker->GetDemuxer();
    auto ret = WriteData(data_ctx, pkt, video_demuxer);
    if (!ret) {
      return ret;
    }

    auto event = std::make_shared<modelbox::FlowUnitEvent>();
    data_ctx->SendEvent(event);
    return modelbox::STATUS_CONTINUE;
  }

  return Reconnect(demux_status, data_ctx);
}

void VideoDemuxerFlowUnit::WriteEnd(
    std::shared_ptr<modelbox::DataContext> &data_ctx) {
  auto demuxer_worker = std::static_pointer_cast<DemuxerWorker>(
      data_ctx->GetPrivate(DEMUXER_CTX));
  auto video_demuxer = demuxer_worker->GetDemuxer();
  auto video_packet_output = data_ctx->Output(VIDEO_PACKET_OUTPUT);
  video_packet_output->Build({1});
  auto end_packet = video_packet_output->At(0);
  int32_t rate_num;
  int32_t rate_den;
  int32_t rotate_angle = video_demuxer->GetFrameRotate();
  video_demuxer->GetFrameRate(rate_num, rate_den);
  end_packet->Set("rate_num", rate_num);
  end_packet->Set("rate_den", rate_den);
  end_packet->Set("rotate_angle", rotate_angle);
  end_packet->Set("duration", video_demuxer->GetDuration());
  end_packet->Set("time_base", video_demuxer->GetTimeBase());
}

modelbox::Status VideoDemuxerFlowUnit::WriteData(
    std::shared_ptr<modelbox::DataContext> &data_ctx,
    std::shared_ptr<AVPacket> &pkt,
    const std::shared_ptr<FfmpegVideoDemuxer> &video_demuxer) {
  if (pkt == nullptr) {
    // no data to send
    return modelbox::STATUS_OK;
  }

  auto video_packet_output = data_ctx->Output(VIDEO_PACKET_OUTPUT);
  std::vector<size_t> shape(1, (size_t)pkt->size);
  if (pkt->size == 0) {
    // Tell decoder end of stream
    video_packet_output->Build({1});
  } else {
    video_packet_output->BuildFromHost(
        shape, pkt->data, pkt->size,
        [pkt](void *ptr) { /* Only capture pkt */ });
  }

  auto packet_buffer = video_packet_output->At(0);
  if (is_retry_reset_) {
    bool is_reset = true;
    auto codec_id = std::make_shared<AVCodecID>(video_demuxer->GetCodecID());
    auto source_url =
        std::static_pointer_cast<std::string>(data_ctx->GetPrivate(SOURCE_URL));
    packet_buffer->Set("reset_flag", is_reset);
    packet_buffer->Set("source_url", *source_url);
    packet_buffer->Set("codec_id", video_demuxer->GetCodecID());
    is_retry_reset_ = false;
  }
  packet_buffer->Set("pts", pkt->pts);
  packet_buffer->Set("dts", pkt->dts);
  packet_buffer->Set("time_base", video_demuxer->GetTimeBase());
  int32_t rate_num;
  int32_t rate_den;
  int32_t frame_width;
  int32_t frame_height;
  int32_t rotate_angle = video_demuxer->GetFrameRotate();
  video_demuxer->GetFrameRate(rate_num, rate_den);
  video_demuxer->GetFrameMeta(&frame_width, &frame_height);
  packet_buffer->Set("rate_num", rate_num);
  packet_buffer->Set("rate_den", rate_den);
  packet_buffer->Set("width", frame_width);
  packet_buffer->Set("height", frame_height);
  packet_buffer->Set("rotate_angle", rotate_angle);
  packet_buffer->Set("duration", video_demuxer->GetDuration());
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoDemuxerFlowUnit::CreateRetryTask(
    std::shared_ptr<modelbox::DataContext> &data_ctx) {
  auto stream_meta = data_ctx->GetInputMeta(STREAM_META_INPUT);
  if (stream_meta == nullptr) {
    return modelbox::STATUS_FAULT;
  }

  auto source_context = std::static_pointer_cast<modelbox::SourceContext>(
      stream_meta->GetMeta(DEMUX_RETRY_CONTEXT));
  if (source_context == nullptr) {
    return modelbox::STATUS_FAULT;
  }

  data_ctx->SetPrivate(DEMUX_RETRY_CONTEXT, source_context);
  source_context->SetLastProcessStatus(modelbox::STATUS_FAULT);
  std::weak_ptr<VideoDemuxerFlowUnit> flowunit = shared_from_this();
  std::weak_ptr<modelbox::DataContext> data_ctx_weak = data_ctx;
  auto timer_task =
      std::make_shared<modelbox::TimerTask>([flowunit, data_ctx_weak]() {
        std::shared_ptr<VideoDemuxerFlowUnit> flow_unit_ = flowunit.lock();
        std::shared_ptr<modelbox::DataContext> data_context =
            data_ctx_weak.lock();
        if (flow_unit_ == nullptr || data_context == nullptr) {
          return;
        }

        auto event = std::make_shared<modelbox::FlowUnitEvent>();
        auto source_context = std::static_pointer_cast<modelbox::SourceContext>(
            data_context->GetPrivate(DEMUX_RETRY_CONTEXT));
        auto source_url = source_context->GetSourceURL();
        modelbox::Status status = modelbox::STATUS_FAULT;
        if (source_url) {
          auto status = flow_unit_->InitDemuxer(data_context, source_url);
        }

        source_context->SetLastProcessStatus(status);
        data_context->SendEvent(event);
      });
  timer_task->SetName("DemuxerReconnect");
  data_ctx->SetPrivate(DEMUX_TIMER_TASK, timer_task);
  return modelbox::STATUS_OK;
}

std::shared_ptr<std::string> VideoDemuxerFlowUnit::GetSourceUrl(
    const std::shared_ptr<modelbox::DataContext> &data_ctx) {
  // Try get url in input meta
  auto stream_meta = data_ctx->GetInputMeta(STREAM_META_INPUT);
  if (stream_meta != nullptr) {
    auto meta_value = stream_meta->GetMeta(SOURCE_URL);
    if (meta_value != nullptr) {
      return std::static_pointer_cast<std::string>(meta_value);
    }
  }

  // Try get url in input buffer
  auto inputs = data_ctx->Input(STREAM_META_INPUT);
  if (inputs == nullptr || inputs->Size() == 0) {
    MBLOG_ERROR << "source url not found in input";
    return nullptr;
  }

  if (inputs->Size() > 1) {
    MBLOG_WARN << "only supports one url for a stream";
  }

  auto input_buffer = inputs->At(0);
  if (input_buffer == nullptr) {
    MBLOG_ERROR << "input buffer for demuxer is nullptr";
    return nullptr;
  }

  return std::make_shared<std::string>(
      (const char *)(input_buffer->ConstData()), input_buffer->GetBytes());
}

modelbox::Status VideoDemuxerFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto source_url_ptr = GetSourceUrl(data_ctx);
  if (source_url_ptr == nullptr) {
    MBLOG_ERROR << "Source url is null, please fill input url correctly";
    return modelbox::STATUS_FAULT;
  }

  auto codec_id = std::make_shared<AVCodecID>();
  auto profile_id = std::make_shared<int32_t>();
  auto source_url = std::make_shared<std::string>();
  auto meta = std::make_shared<modelbox::DataMeta>();
  meta->SetMeta(CODEC_META, codec_id);
  meta->SetMeta(PROFILE_META, profile_id);
  meta->SetMeta(SOURCE_URL, source_url);
  data_ctx->SetOutputMeta(VIDEO_PACKET_OUTPUT, meta);
  data_ctx->SetPrivate(VIDEO_PACKET_OUTPUT, meta);

  auto demuxer_status = InitDemuxer(data_ctx, source_url_ptr);

  if (demuxer_status != modelbox::STATUS_OK) {
    MBLOG_INFO << "failed init Demuxer";
  }

  auto ret = CreateRetryTask(data_ctx);
  if (!ret && !demuxer_status) {
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_SUCCESS;
}

void VideoDemuxerFlowUnit::UpdateStatsInfo(
    const std::shared_ptr<modelbox::DataContext> &data_ctx,
    const std::shared_ptr<FfmpegVideoDemuxer> &demuxer) {
  auto stats = data_ctx->GetStatistics();
  int32_t frame_rate_num = 0;
  int32_t frame_rate_den = 0;
  demuxer->GetFrameRate(frame_rate_num, frame_rate_den);
  stats->AddItem("frame_rate_num", frame_rate_num, true);
  stats->AddItem("frame_rate_den", frame_rate_den, true);
}

modelbox::Status VideoDemuxerFlowUnit::InitDemuxer(
    std::shared_ptr<modelbox::DataContext> &data_ctx,
    std::shared_ptr<std::string> &source_url) {
  auto reader = std::make_shared<FfmpegReader>();
  auto ret = reader->Open(*source_url);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_INFO << "Open reader falied, set DEMUX_STATUS failed";
    return modelbox::STATUS_FAULT;
  }

  auto video_demuxer = std::make_shared<FfmpegVideoDemuxer>();
  ret = video_demuxer->Init(reader, key_frame_only_);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_INFO << "video demux init falied, set DEMUX_STATUS failed";
    return modelbox::STATUS_FAULT;
  }
  video_demuxer->LogStreamInfo();

  int32_t width = 0;
  int32_t height = 0;
  video_demuxer->GetFrameMeta(&width, &height);
  if (width == 0 || height == 0) {
    MBLOG_ERROR << "video demuxer get frame meta failed";
    return modelbox::STATUS_FAULT;
  }

  auto codec_id = video_demuxer->GetCodecID();
  auto profile_id = video_demuxer->GetProfileID();
  // reset meta value
  auto meta = std::static_pointer_cast<modelbox::DataMeta>(
      data_ctx->GetPrivate(VIDEO_PACKET_OUTPUT));
  auto code_meta = std::static_pointer_cast<int>(meta->GetMeta(CODEC_META));
  *code_meta = codec_id;
  auto profile_meta =
      std::static_pointer_cast<int>(meta->GetMeta(PROFILE_META));
  *profile_meta = profile_id;
  auto uri_meta =
      std::static_pointer_cast<std::string>(meta->GetMeta(SOURCE_URL));
  *uri_meta = *source_url;

  auto is_rtsp = (source_url->find("rtsp://") == 0);
  auto demuxer_worker =
      std::make_shared<DemuxerWorker>(is_rtsp, queue_size_, video_demuxer);
  ret = demuxer_worker->Init();
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "init demuxer failed, ret " << ret;
    return ret;
  }

  data_ctx->SetPrivate(DEMUXER_CTX, demuxer_worker);
  data_ctx->SetPrivate(SOURCE_URL, source_url);

  UpdateStatsInfo(data_ctx, video_demuxer);
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoDemuxerFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto timer_task = std::static_pointer_cast<modelbox::TimerTask>(
      data_ctx->GetPrivate(DEMUX_TIMER_TASK));

  if (timer_task) {
    timer_task->Stop();
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(VideoDemuxerFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Video");
  desc.AddFlowUnitInput({STREAM_META_INPUT});
  desc.AddFlowUnitOutput({VIDEO_PACKET_OUTPUT});
  desc.SetFlowType(modelbox::FlowType::STREAM);
  desc.SetStreamSameCount(false);
  desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}

DemuxerWorker::DemuxerWorker(bool is_async, size_t cache_size,
                             std::shared_ptr<FfmpegVideoDemuxer> demuxer)
    : is_async_(is_async),
      cache_size_(cache_size),
      demuxer_(std::move(demuxer)) {
  const size_t min_cache_size = 32;
  if (cache_size_ < min_cache_size) {
    cache_size_ = min_cache_size;
  }
}

DemuxerWorker::~DemuxerWorker() {
  if (demux_thread_ != nullptr) {
    demux_thread_running_ = false;
    demux_thread_->join();
  }
}

modelbox::Status DemuxerWorker::Init() {
  if (!is_async_) {
    return modelbox::STATUS_OK;
  }

  demux_thread_running_ = true;
  demux_thread_ = std::make_shared<std::thread>([this]() {
    while (IsRunning()) {
      Process();
    }
  });

  return modelbox::STATUS_OK;
}

std::shared_ptr<FfmpegVideoDemuxer> DemuxerWorker::GetDemuxer() const {
  return demuxer_;
}

size_t DemuxerWorker::GetDropCount() const { return packet_drop_count_; }

modelbox::Status DemuxerWorker::ReadPacket(
    std::shared_ptr<AVPacket> &av_packet) {
  if (!is_async_) {
    return demuxer_->Demux(av_packet);
  }

  auto ret = PopCache(av_packet);
  if (ret != modelbox::STATUS_OK) {
    // demuxer read end
    return last_demux_status_;
  }

  return modelbox::STATUS_OK;
}

bool DemuxerWorker::IsRunning() const { return demux_thread_running_; }

void DemuxerWorker::Process() {
  std::shared_ptr<AVPacket> av_packet;
  last_demux_status_ = demuxer_->Demux(av_packet);
  if (last_demux_status_ != modelbox::STATUS_OK) {
    demux_thread_running_ = false;
    av_packet = nullptr;
    std::unique_lock<std::mutex> lock(packet_cache_lock_);
    packet_cache_.push_back(av_packet);
    packet_cache_not_empty_.notify_all();
    return;
  }

  PushCache(av_packet);
}

void DemuxerWorker::PushCache(const std::shared_ptr<AVPacket> &av_packet) {
  std::unique_lock<std::mutex> lock(packet_cache_lock_);
  if (missing_pre_packet_) {
    if (!IsKeyFrame(av_packet)) {
      // not key frame, continue drop this packet
      ++packet_drop_count_;
      return;
    }

    // this packet is key frame, push to cache, continue decode
    missing_pre_packet_ = false;
    packet_cache_.push_back(av_packet);
    packet_cache_not_empty_.notify_all();
    return;
  }

  if (packet_cache_.size() >= cache_size_) {
    // need drop packet in cache
    do {
      // drop front until key frame
      packet_cache_.pop_front();
      ++packet_drop_count_;
      if (!packet_cache_.empty()) {
        continue;
      }

      // all cache dropped
      if (!IsKeyFrame(av_packet)) {
        // not key frame, drop this packet too
        // set flag to wait next key frame
        missing_pre_packet_ = true;
        ++packet_drop_count_;
        return;
      }

      // this is key frame, push to cache
      break;
    } while (!IsKeyFrame(packet_cache_.front()));

    // find key frame, push this packet to cache
  }

  // push this packet to cache
  packet_cache_.push_back(av_packet);
  packet_cache_not_empty_.notify_all();
}

modelbox::Status DemuxerWorker::PopCache(std::shared_ptr<AVPacket> &av_packet) {
  std::unique_lock<std::mutex> lock(packet_cache_lock_);
  packet_cache_not_empty_.wait_for(lock, std::chrono::milliseconds(20),
                                   [&]() { return !packet_cache_.empty(); });
  if (packet_cache_.empty()) {
    // avoid to stuck other stream in node::run, we need return when
    // packet_cache has no data
    av_packet = nullptr;
    return modelbox::STATUS_OK;
  }

  av_packet = packet_cache_.front();
  if (av_packet == nullptr) {
    // stream end, keep nullptr in cache
    return modelbox::STATUS_NODATA;
  }

  packet_cache_.pop_front();
  return modelbox::STATUS_OK;
}

bool DemuxerWorker::IsKeyFrame(const std::shared_ptr<AVPacket> &av_packet) {
  return (av_packet->flags & AV_PKT_FLAG_KEY) != 0;
}
