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

#include "video_out_flowunit.h"

#include <securec.h>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <regex>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

modelbox::Status VideoOutFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  default_dest_url_ = opts->GetString("default_dest_url", "");

  return modelbox::STATUS_OK;
}

modelbox::Status VideoOutFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status VideoOutFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto image_queue = std::static_pointer_cast<
      modelbox::BlockingQueue<std::shared_ptr<modelbox::Buffer>>>(
      ctx->GetPrivate(SHOW_QUEUE_CTX));
  if (image_queue != nullptr) {
    auto input_buffer_list = ctx->Input(FRAME_INFO_INPUT);
    for (size_t i = 0; i < input_buffer_list->Size(); ++i) {
      image_queue->Push(input_buffer_list->At(i), 50);
    }
    return modelbox::STATUS_SUCCESS;
  }

  // others do video encoding
  auto muxer =
      std::static_pointer_cast<FfmpegVideoMuxer>(ctx->GetPrivate(MUXER_CTX));
  auto encoder = std::static_pointer_cast<FfmpegVideoEncoder>(
      ctx->GetPrivate(ENCODER_CTX));
  if (muxer == nullptr || encoder == nullptr) {
    MBLOG_ERROR << "Stream not inited";
    return {modelbox::STATUS_FAULT, "Stream not inited"};
  }

  std::vector<std::shared_ptr<AVFrame>> av_frame_list;
  auto ret = ReadFrames(ctx, av_frame_list);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Read input frame failed";
    return {modelbox::STATUS_FAULT, "Read input frame failed"};
  }

  std::vector<std::shared_ptr<AVPacket>> av_packet_list;
  ret = EncodeFrame(encoder, av_frame_list, av_packet_list);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Encode frame failed";
    return {modelbox::STATUS_FAULT, "Encode frame failed"};
  }

  ret = MuxPacket(muxer, encoder->GetCtx()->time_base, av_packet_list);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Mux packet failed";
    return {modelbox::STATUS_FAULT, "Mux packet failed"};
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoOutFlowUnit::ReadFrames(
    const std::shared_ptr<modelbox::DataContext> &ctx,
    std::vector<std::shared_ptr<AVFrame>> &av_frame_list) {
  auto frame_buffer_list = ctx->Input(FRAME_INFO_INPUT);
  if (frame_buffer_list == nullptr || frame_buffer_list->Size() == 0) {
    MBLOG_ERROR << "Input frame list is empty";
    return {modelbox::STATUS_FAULT, "Input frame list is empty"};
  }

  auto frame_index_ptr =
      std::static_pointer_cast<int64_t>(ctx->GetPrivate(FRAME_INDEX_CTX));
  for (auto frame_buffer : *frame_buffer_list) {
    std::shared_ptr<AVFrame> av_frame;
    auto ret = ReadFrameFromBuffer(frame_buffer, av_frame);
    av_frame->pts = *frame_index_ptr;
    ++(*frame_index_ptr);
    if (ret != modelbox::STATUS_SUCCESS) {
      MBLOG_ERROR << "Read frame from buffer failed";
      return ret;
    }

    av_frame_list.push_back(av_frame);
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoOutFlowUnit::ReadFrameFromBuffer(
    std::shared_ptr<modelbox::Buffer> &frame_buffer,
    std::shared_ptr<AVFrame> &av_frame) {
  auto *frame_ptr = av_frame_alloc();
  if (frame_ptr == nullptr) {
    MBLOG_ERROR << "Alloca frame failed";
    return {modelbox::STATUS_FAULT, "Alloca frame failed"};
  }

  av_frame.reset(frame_ptr, [](AVFrame *ptr) { av_frame_free(&ptr); });
  frame_buffer->Get("width", av_frame->width);
  frame_buffer->Get("height", av_frame->height);
  std::string pix_fmt;
  frame_buffer->Get("pix_fmt", pix_fmt);
  auto iter = videodecode::g_av_pix_fmt_map.find(pix_fmt);
  if (iter == videodecode::g_av_pix_fmt_map.end()) {
    MBLOG_ERROR << "Encoder not support pix fmt " << pix_fmt;
    return {modelbox::STATUS_NOTSUPPORT,
            "Encoder not support pix fmt " + pix_fmt};
  }
  av_frame->format = iter->second;
  auto ret =
      av_image_fill_arrays(av_frame->data, av_frame->linesize,
                           (const uint8_t *)frame_buffer->ConstData(),
                           iter->second, av_frame->width, av_frame->height, 1);
  if (ret < 0) {
    GET_FFMPEG_ERR(ret, ffmpeg_err);
    MBLOG_ERROR << "avpicture_fill failed, err " << ffmpeg_err;
    return {modelbox::STATUS_FAULT, "avpicture_fill failed, err "};
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoOutFlowUnit::EncodeFrame(
    const std::shared_ptr<FfmpegVideoEncoder> &encoder,
    const std::vector<std::shared_ptr<AVFrame>> &av_frame_list,
    std::vector<std::shared_ptr<AVPacket>> &av_packet_list) {
  for (const auto &frame : av_frame_list) {
    auto ret = encoder->Encode(GetBindDevice(), frame, av_packet_list);
    if (ret != modelbox::STATUS_SUCCESS) {
      MBLOG_ERROR << "Encoder encode frame failed reason: " + ret.Errormsg();
      return ret;
    }
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoOutFlowUnit::MuxPacket(
    const std::shared_ptr<FfmpegVideoMuxer> &muxer, const AVRational &time_base,
    std::vector<std::shared_ptr<AVPacket>> &av_packet_list) {
  for (const auto &packet : av_packet_list) {
    auto ret = muxer->Mux(time_base, packet);
    if (ret != modelbox::STATUS_SUCCESS) {
      MBLOG_ERROR << "Muxer mux packet failed";
      return ret;
    }
  }

  return modelbox::STATUS_SUCCESS;
}

void VideoOutFlowUnit::ProcessShow(
    const std::string &dest_url,
    const std::shared_ptr<
        modelbox::BlockingQueue<std::shared_ptr<modelbox::Buffer>>>
        &image_queue) {
  std::string win_name = "modelbox_show";
  if (dest_url.length() > 2) {
    win_name = dest_url.substr(2);
  }

  cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
  std::shared_ptr<modelbox::Buffer> buf;
  std::shared_ptr<modelbox::Buffer> back_buf;
  while (image_queue->Pop(&buf)) {
    if (buf == nullptr) {
      break;
    }

    // at least 1, even not set widht, height
    int32_t width = 1;
    int32_t height = 1;
    std::string pix_fmt = "bgr";
    buf->Get("width", width);
    buf->Get("height", height);
    buf->Get("pix_fmt", pix_fmt);
    void *input_data = const_cast<void *>(buf->ConstData());
    bool isnv12 = (pix_fmt == "nv12");
    cv::Mat img_data(cv::Size(width, isnv12 ? height * 3 / 2 : height),
                     isnv12 ? CV_8UC1 : CV_8UC3, input_data);
    cv::Mat show_img = img_data;
    // todo color change
    if (pix_fmt == "rgb") {
      cv::cvtColor(img_data, show_img, cv::COLOR_RGB2BGR);
    } else if (pix_fmt == "nv12") {
      cv::cvtColor(img_data, show_img, cv::COLOR_YUV2BGR_NV12);
    }

    cv::imshow(win_name, show_img);
    cv::waitKey(10);
    back_buf = buf;
  }

  cv::destroyWindow(win_name);
}

modelbox::Status VideoOutFlowUnit::PrepareVideoOut(
    const std::shared_ptr<modelbox::DataContext> &data_ctx,
    const std::string &dest_url, const std::string &format_name) {
  auto frame_buffer_list = data_ctx->Input(FRAME_INFO_INPUT);
  if (frame_buffer_list == nullptr || frame_buffer_list->Size() == 0) {
    MBLOG_ERROR << "Input [frame_info] is empty";
    return {modelbox::STATUS_FAULT, "Input [frame_info] is empty"};
  }

  auto frame_buffer = frame_buffer_list->At(0);
  int32_t width = 0;
  int32_t height = 0;
  int32_t rate_num = 25;
  int32_t rate_den = 1;
  frame_buffer->Get("width", width);
  frame_buffer->Get("height", height);
  frame_buffer->Get("rate_num", rate_num);
  frame_buffer->Get("rate_den", rate_den);

  if (width == 0 || height == 0) {
    MBLOG_ERROR << "buffer meta is invalid";
    return {modelbox::STATUS_INVALID, "buffer meta is invalid"};
  }

  auto encoder = std::make_shared<FfmpegVideoEncoder>();
  auto ret =
      encoder->Init(GetBindDevice(), width, height, {rate_num, rate_den});
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Init encoder failed";
    return {modelbox::STATUS_FAULT, "Init encoder failed"};
  }

  auto writer = std::make_shared<FfmpegWriter>();
  ret = writer->Open(format_name, dest_url);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Open ffmepg writer failed, format " << format_name
                << ", url " << dest_url;
    return {modelbox::STATUS_FAULT, "Open ffmepg writer failed, format " +
                                        format_name + ", url " + dest_url};
  }

  auto muxer = std::make_shared<FfmpegVideoMuxer>();
  ret = muxer->Init(encoder->GetCtx(), writer);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Init muxer failed";
    return {modelbox::STATUS_FAULT, "Init muxer failed"};
  }

  auto color_cvt = std::make_shared<FfmpegColorConverter>();

  data_ctx->SetPrivate(MUXER_CTX, muxer);
  data_ctx->SetPrivate(ENCODER_CTX, encoder);
  data_ctx->SetPrivate(COLOR_CVT_CTX, color_cvt);
  return modelbox::STATUS_OK;
}

modelbox::Status VideoOutFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  std::string dest_url;
  auto ret = GetDestUrl(data_ctx, dest_url);
  if (ret != modelbox::STATUS_SUCCESS || dest_url.empty()) {
    MBLOG_ERROR << "dest_url in config is empty, no dest url available";
    return {modelbox::STATUS_FAULT,
            "dest_url in config is empty, no dest url available"};
  }

  MBLOG_INFO << "videoout url=" << dest_url;

  auto frame_index_ptr = std::make_shared<int64_t>(0);
  data_ctx->SetPrivate(FRAME_INDEX_CTX, frame_index_ptr);

  if (dest_url[0] >= '0' && dest_url[0] <= '9') {
    // 视频输出， 类似0:windows_name配置
    std::shared_ptr<std::thread> show_thread;
    auto image_queue = std::make_shared<
        modelbox::BlockingQueue<std::shared_ptr<modelbox::Buffer>>>(2);
    show_thread.reset(new std::thread(&VideoOutFlowUnit::ProcessShow, this,
                                      dest_url, image_queue),
                      [image_queue](std::thread *p) {
                        image_queue->Shutdown();
                        if (p && p->joinable()) {
                          p->join();
                        }
                        delete p;
                      });
    data_ctx->SetPrivate(SHOW_CTX, show_thread);
    data_ctx->SetPrivate(SHOW_QUEUE_CTX, image_queue);
    return modelbox::STATUS_OK;
  }

  std::string format_name = "mp4";
  if (dest_url.substr(0, 4) == "rtsp") {
    format_name = "rtsp";
  }

  return PrepareVideoOut(data_ctx, dest_url, format_name);
}

modelbox::Status VideoOutFlowUnit::GetDestUrl(
    const std::shared_ptr<modelbox::DataContext> &data_ctx,
    std::string &dest_url) {
  dest_url = default_dest_url_;
  Defer {
    std::regex url_auth_pattern("://[^ /]*?:[^ /]*?@");
    auto result = std::regex_replace(dest_url, url_auth_pattern, "://*:*@");
    MBLOG_INFO << "video_out url is " << result;
  };

  if (data_ctx == nullptr) {
    MBLOG_ERROR << "data ctx is nullptr";
    return modelbox::STATUS_INVALID;
  }

  // 3种方式获取
  auto stream_meta = data_ctx->GetInputMeta(FRAME_INFO_INPUT);
  if (stream_meta != nullptr) {
    auto meta_dest_url = stream_meta->GetMeta(DEST_URL);
    auto dest_url_ptr =
        meta_dest_url == nullptr
            ? nullptr
            : std::static_pointer_cast<std::string>(meta_dest_url);
    if (dest_url_ptr != nullptr && !(*dest_url_ptr).empty()) {
      dest_url = *dest_url_ptr;
      return modelbox::STATUS_SUCCESS;
    }
  }

  auto config = data_ctx->GetSessionConfig();
  if (config == nullptr) {
    MBLOG_ERROR << "data ctx session config is empty";
    return modelbox::STATUS_INVALID;
  }

  auto cfg_str = config->GetString("iva_task_output");
  if (cfg_str.empty()) {
    return modelbox::STATUS_SUCCESS;
  }

  try {
    nlohmann::json url_json = nlohmann::json::parse(cfg_str);
    if (url_json.contains("data") && url_json["data"].contains("url")) {
      dest_url = url_json["data"]["url"].get<std::string>();
    }
  } catch (const std::exception &e) {
    MBLOG_ERROR << "Parse config str to json failed, detail: " << e.what();
    return modelbox::STATUS_INVALID;
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoOutFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  data_ctx->SetPrivate(MUXER_CTX, nullptr);
  data_ctx->SetPrivate(ENCODER_CTX, nullptr);
  data_ctx->SetPrivate(SHOW_CTX, nullptr);
  data_ctx->SetPrivate(SHOW_QUEUE_CTX, nullptr);
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(VideoOutFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Video");
  desc.AddFlowUnitInput({FRAME_INFO_INPUT, "cpu"});
  desc.SetFlowType(modelbox::STREAM);
  // 禁止异步执行，编码必须一帧帧的编码
  desc.SetResourceNice(false);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "default_dest_url", "string", true, "", "the encoder dest url"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
