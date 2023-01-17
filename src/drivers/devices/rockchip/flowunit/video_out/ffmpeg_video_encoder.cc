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

#include "ffmpeg_video_encoder.h"

#include <modelbox/base/log.h>

#include "modelbox/device/rockchip/rockchip_memory.h"
#include "securec.h"

#define ENCODER_PROFILE 66
#define RK_POLL_TIMEOUT 500

modelbox::Status FfmpegVideoEncoder::Init_PrepConfig() {
  MppEncPrepCfg prep_cfg;
  (void)memset_s(&prep_cfg, sizeof(MppEncPrepCfg), 0, sizeof(MppEncPrepCfg));

  prep_cfg.change = MPP_ENC_PREP_CFG_CHANGE_INPUT |
                    MPP_ENC_PREP_CFG_CHANGE_ROTATION |
                    MPP_ENC_PREP_CFG_CHANGE_FORMAT;
  prep_cfg.width = width_;
  prep_cfg.height = height_;
  prep_cfg.hor_stride = alignW_;
  prep_cfg.ver_stride = alignH_;
  prep_cfg.format = MPP_FMT_YUV420SP;
  prep_cfg.rotation = MPP_ENC_ROT_0;
  auto ret = rk_api_->control(codec_ctx_, MPP_ENC_SET_PREP_CFG, &prep_cfg);
  if (ret) {
    MBLOG_ERROR << "mpi control enc set prep cfg failed ret=" << ret;
    return {modelbox::STATUS_FAULT, "mpi control enc set prep cfg failed"};
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status FfmpegVideoEncoder::Init_RcConfig() {
  MppEncRcCfg rc_cfg = {0};
  (void)memset_s(&rc_cfg, sizeof(MppEncRcCfg), 0, sizeof(MppEncRcCfg));

  rc_cfg.change = MPP_ENC_RC_CFG_CHANGE_ALL;
  rc_cfg.rc_mode = MPP_ENC_RC_MODE_AVBR;
  rc_cfg.quality = MPP_ENC_RC_QUALITY_MEDIUM;

  int wh = width_ * height_;
  if (wh <= 640 * 480) {
    bps_ = (int)(wh * 3.5);
  } else {
    bps_ = (int)(wh * (codec_type_ == MPP_VIDEO_CodingHEVC ? 2.5 : 3));
  }

  if (rc_cfg.rc_mode == MPP_ENC_RC_MODE_CBR) {
    /* constant bitrate has very small bps_ range of 1/16 bps_ */
    rc_cfg.bps_target = bps_;
    rc_cfg.bps_max = bps_ * 17 / 16;
    rc_cfg.bps_min = bps_ * 15 / 16;
  } else if (rc_cfg.rc_mode == MPP_ENC_RC_MODE_AVBR) {
    /* variable bitrate has large bps_ range */
    rc_cfg.bps_target = bps_;
    rc_cfg.bps_max = bps_ * 17 / 16;
    rc_cfg.bps_min = bps_ * 1 / 16;
  }

  /* fix input / output frame rate */
  rc_cfg.fps_in_flex = 0;
  rc_cfg.fps_in_num = fps_;
  rc_cfg.fps_in_denorm = fps_den_;
  rc_cfg.fps_out_flex = 0;
  rc_cfg.fps_out_num = fps_;
  rc_cfg.fps_out_denorm = fps_den_;

  auto fgop = fps_ * 1.0 / fps_den_ + 0.5;
  rc_cfg.gop = (int)(fgop)*2;
  rc_cfg.max_reenc_times = 0;

  int qp_init = 26;
  int qp_max = 0;
  int qp_min = 0;
  int qp_step = 0;

  if (rc_cfg.rc_mode == MPP_ENC_RC_MODE_CBR) {
    /* constant bitrate do not limit qp range */
    qp_max = 48;
    qp_min = 4;
    qp_step = 16;
    qp_init = 0;
  } else if (rc_cfg.rc_mode == MPP_ENC_RC_MODE_AVBR) {
    /* variable bitrate has qp min limit */
    qp_max = 48;
    qp_min = 10;
    qp_step = 4;
    qp_init = 10;
  }

  rc_cfg.qp_max = qp_max;
  rc_cfg.qp_min = qp_min;
  rc_cfg.qp_max_i = qp_max;
  rc_cfg.qp_min_i = qp_min;
  rc_cfg.qp_init = qp_init;
  rc_cfg.qp_max_step = qp_step;
  rc_cfg.qp_delta_ip = 4;
  rc_cfg.qp_delta_vi = 2;

  auto ret = rk_api_->control(codec_ctx_, MPP_ENC_SET_RC_CFG, &rc_cfg);
  if (ret) {
    MBLOG_ERROR << "mpi control enc set rc cfg failed ret=" << ret;
    return {modelbox::STATUS_FAULT, "mpi control enc set rc cfg failed"};
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status FfmpegVideoEncoder::Init_CodecConfig() {
  MppEncCodecCfg codec_cfg;
  (void)memset_s(&codec_cfg, sizeof(MppEncCodecCfg), 0, sizeof(MppEncCodecCfg));

  codec_cfg.coding = codec_type_;
  switch (codec_cfg.coding) {
    case MPP_VIDEO_CodingAVC: {
      codec_cfg.h264.change =
          MPP_ENC_H264_CFG_CHANGE_PROFILE | MPP_ENC_H264_CFG_CHANGE_ENTROPY |
          MPP_ENC_H264_CFG_CHANGE_TRANS_8x8 | MPP_ENC_H264_CFG_CHANGE_QP_LIMIT;

      codec_cfg.h264.profile = ENCODER_PROFILE;
      codec_cfg.h264.level = (width_ > 1280) ? 40 : 31;
      codec_cfg.h264.entropy_coding_mode = 0;  // baseline=0, others=1
      codec_cfg.h264.cabac_init_idc = 0;
      codec_cfg.h264.transform8x8_mode = 0;  // baseline=0, others=1
    } break;
    case MPP_VIDEO_CodingMJPEG:
    case MPP_VIDEO_CodingHEVC:
    case MPP_VIDEO_CodingVP8:
    default: {
      auto msg = std::string("unsupport encoder coding type =") +
                 std::to_string(codec_cfg.coding);
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    } break;
  }

  auto ret = rk_api_->control(codec_ctx_, MPP_ENC_SET_CODEC_CFG, &codec_cfg);
  if (ret) {
    MBLOG_ERROR << "mpi control enc set codec cfg failed ret=" << ret;
    return {modelbox::STATUS_FAULT, "mpi control enc set codec cfg failed"};
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status FfmpegVideoEncoder::Init_Config() {
  auto ret = mpp_enc_cfg_init(&cfg_);
  if (ret != MPP_OK) {
    MBLOG_ERROR << "mpi control enc get cfg failed ret=" << ret;
    return {modelbox::STATUS_FAULT, "mpi control enc get cfg failed"};
  }

  auto ret_config = Init_PrepConfig();
  if (ret_config != modelbox::STATUS_SUCCESS) {
    return ret_config;
  }

  ret_config = Init_RcConfig();
  if (ret_config != modelbox::STATUS_SUCCESS) {
    return ret_config;
  }

  ret_config = Init_CodecConfig();
  if (ret_config != modelbox::STATUS_SUCCESS) {
    return ret_config;
  }

  /* optional */
  int sei_mode = MPP_ENC_SEI_MODE_ONE_FRAME;
  ret = rk_api_->control(codec_ctx_, MPP_ENC_SET_SEI_CFG, &sei_mode);
  if (ret != MPP_OK) {
    MBLOG_ERROR << "mpi control enc set sei cfg failed ret=" << ret;
    return {modelbox::STATUS_FAULT, "mpi control enc set sei cfg failed"};
  }

  int header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
  ret = rk_api_->control(codec_ctx_, MPP_ENC_SET_HEADER_MODE, &header_mode);
  if (ret != MPP_OK) {
    MBLOG_ERROR << "mpi control enc set header mode failed ret=" << ret;
    return {modelbox::STATUS_FAULT, "mpi control enc set header mode failed"};
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status FfmpegVideoEncoder::Init_MppContex() {
  auto ret = mpp_create(&codec_ctx_, &rk_api_);
  if (ret != MPP_OK) {
    MBLOG_ERROR << "failed to run mpp_create: " << ret;
    return {modelbox::STATUS_FAULT, "failed to run mpp_create"};
  }

  RK_U32 timeout = RK_POLL_TIMEOUT;
  ret = rk_api_->control(codec_ctx_, MPP_SET_OUTPUT_TIMEOUT, &timeout);
  if (ret != MPP_OK) {
    MBLOG_ERROR << "mpi control set output timeout ret=" << ret;
    return {modelbox::STATUS_FAULT, "mpi control set output timeout"};
  }

  ret = mpp_init(codec_ctx_, MPP_CTX_ENC, codec_type_);
  if (ret != MPP_OK) {
    MBLOG_ERROR << "mpp_init failed ret=" << ret;
    return {modelbox::STATUS_FAULT, "mpp_init failed"};
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status FfmpegVideoEncoder::RkInit(int w, int h,
                                            const AVRational &frame_rate,
                                            const std::string &encodeType) {
  codec_type_ = MPP_VIDEO_CodingAVC;
  width_ = w;
  height_ = h;
  fps_ = frame_rate.num;
  fps_den_ = frame_rate.den;

  alignW_ = MPP_ALIGN(w, MPP_ALIGN_WIDTH);
  alignH_ = MPP_ALIGN(h, MPP_ALIGN_HEIGHT);

  auto ret = Init_MppContex();
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "failed to init mpp contex reason: " << ret.Errormsg();
    return {modelbox::STATUS_FAULT,
            "failed to init mpp contex reason: " + ret.Errormsg()};
  }

  ret = Init_Config();
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "failed to init config reason: " << ret.Errormsg();
    return {modelbox::STATUS_FAULT,
            "failed to init config reason: " + ret.Errormsg()};
  }

  return modelbox::STATUS_SUCCESS;
}

void FfmpegVideoEncoder::CloseRkEncoder() {
  if (cfg_) {
    mpp_enc_cfg_deinit(cfg_);  // todo: here or Init_Config?
  }
  if (rk_api_ && codec_ctx_) {
    rk_api_->reset(codec_ctx_);
  }

  if (codec_ctx_) {
    mpp_destroy(codec_ctx_);
    codec_ctx_ = nullptr;
  }
}

modelbox::Status FfmpegVideoEncoder::Init(
    const std::shared_ptr<modelbox::Device> &device, int32_t width,
    int32_t height, const AVRational &frame_rate) {
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 9, 100)
  av_register_all();
#endif

  std::string encoder_name = "h264";
  auto *av_codec_ctx = (AVCodecContext *)av_malloc(sizeof(AVCodecContext));
  (void)memset_s(av_codec_ctx, sizeof(AVCodecContext), 0,
                 sizeof(AVCodecContext));
  if (av_codec_ctx == nullptr) {
    MBLOG_ERROR << "Alloc codec ctx failed, encoder name:" << encoder_name;
    return {modelbox::STATUS_FAULT,
            "Alloc codec ctx failed, encoder name:" + encoder_name};
  }

  av_codec_ctx_.reset(av_codec_ctx, [this](AVCodecContext *ctx) {
    if (ctx->extradata) {
      av_free(ctx->extradata);
    }

    av_free(ctx);
    CloseRkEncoder();
  });

  auto ret = RkInit(width, height, frame_rate, encoder_name);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "rk init fail";
    return {modelbox::STATUS_FAULT,
            "failed to rk init reason: " + ret.Errormsg()};
  }

  SetupCodecParam(width, height, frame_rate, av_codec_ctx_);
  // add extra data
  auto buffer = std::make_shared<modelbox::Buffer>(device);
  buffer->Build(500);
  MppPacket packet = nullptr;
  mpp_packet_init_with_buffer(&packet, (MppBuffer)(buffer->MutableData()));
  /* NOTE: It is important to clear output packet length!! */
  mpp_packet_set_length(packet, 0);

  auto mppret = rk_api_->control(codec_ctx_, MPP_ENC_GET_HDR_SYNC, packet);
  if (mppret == MPP_OK) {
    void *ptr = mpp_packet_get_pos(packet);
    av_codec_ctx_->extradata_size = (int)(mpp_packet_get_length(packet));
    av_codec_ctx_->extradata =
        (uint8_t *)av_malloc(av_codec_ctx_->extradata_size);
    if (av_codec_ctx_->extradata) {
      (void)memcpy_s(av_codec_ctx_->extradata, av_codec_ctx_->extradata_size,
                     ptr, av_codec_ctx_->extradata_size);
    }
  }

  mpp_packet_set_buffer(packet, nullptr);
  mpp_packet_deinit(&packet);

  return modelbox::STATUS_SUCCESS;
}

void FfmpegVideoEncoder::SetupCodecParam(
    int32_t width, int32_t height, const AVRational &frame_rate,
    std::shared_ptr<AVCodecContext> &codec_ctx) {
  codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
  codec_ctx->codec_id = (codec_type_ == MPP_VIDEO_CodingAVC) ? AV_CODEC_ID_H264
                                                             : AV_CODEC_ID_HEVC;
  codec_ctx->bit_rate = bps_;
  codec_ctx->profile = ENCODER_PROFILE;
  codec_ctx->level = (width > 1280) ? 40 : 31;

  codec_ctx->pix_fmt = AV_PIX_FMT_NV12;
  codec_ctx->width = width;
  codec_ctx->height = height;
  codec_ctx->color_primaries = AVCOL_PRI_UNSPECIFIED;
  codec_ctx->color_trc = AVCOL_TRC_UNSPECIFIED;
  codec_ctx->colorspace = AVCOL_SPC_UNSPECIFIED;
  codec_ctx->chroma_sample_location = AVCHROMA_LOC_UNSPECIFIED;
  codec_ctx->sample_aspect_ratio = {0, 1};
  codec_ctx->has_b_frames = 0;

  codec_ctx->framerate = frame_rate;
  codec_ctx->time_base = av_inv_q(frame_rate);
  codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
}

std::shared_ptr<AVPacket> FfmpegVideoEncoder::NewPacket(MppPacket &packet) {
  if (packet == nullptr) {
    MBLOG_ERROR << "packet is nullptr";
    return nullptr;
  }

  void *ptr = mpp_packet_get_pos(packet);
  size_t len = mpp_packet_get_length(packet);

  auto *av_packet_ptr = av_packet_alloc();

  Defer { mpp_packet_deinit(&packet); };
  if (av_packet_ptr == nullptr) {
    MBLOG_ERROR << "av packet alloc failed";
    return nullptr;
  }
  if (0 != av_new_packet(av_packet_ptr, (int)len)) {
    MBLOG_ERROR << "av packet new failed";
    return nullptr;
  }

  std::shared_ptr<AVPacket> av_packet(
      av_packet_ptr, [](AVPacket *pkt) { av_packet_free(&pkt); });

  if (len > 0) {
    auto e_ret = memcpy_s(av_packet_ptr->data, len, ptr, len);
    if (e_ret != EOK) {
      MBLOG_ERROR << "av packet memcpy_s failed ret: " << e_ret;
      return nullptr;
    }
  } else {
    MBLOG_WARN << "get one zero compress frame";
  }

  av_packet_ptr->pts = mpp_packet_get_pts(packet);
  av_packet_ptr->dts = av_packet_ptr->pts;

  return av_packet;
}

std::shared_ptr<modelbox::Buffer> FfmpegVideoEncoder::FromAvFrame(
    const std::shared_ptr<modelbox::Device> &device,
    const std::shared_ptr<AVFrame> &av_frame) {
  size_t w = av_frame->width;
  size_t h = av_frame->height;
  size_t total_size = av_frame->format == AVPixelFormat::AV_PIX_FMT_NV12
                          ? w * h * 3 / 2
                          : w * h * 3;
  // assume buffer is allignmented

  auto buffer = std::make_shared<modelbox::Buffer>(device);
  buffer->Build(total_size);
  auto *mpp_buf = (MppBuffer)(buffer->MutableData());
  auto *cpu_buf = (uint8_t *)mpp_buffer_get_ptr(mpp_buf);

  auto ret = av_image_copy_to_buffer(
      cpu_buf, (int)total_size, av_frame->data, av_frame->linesize,
      (AVPixelFormat)(av_frame->format), w, h, 1);
  if (ret < 0) {
    MBLOG_ERROR << "failed to av_image_copy_to_buffer: " << ret;
    return nullptr;
  }

  if (av_frame->format != AVPixelFormat::AV_PIX_FMT_NV12) {
    MppFrame frame = nullptr;
    auto ret = mpp_frame_init(&frame);
    if (ret != MPP_OK) {
      MBLOG_ERROR << "FromAvFrame frame failed ";
      return nullptr;
    }

    mpp_frame_set_width(frame, w);
    mpp_frame_set_height(frame, h);
    mpp_frame_set_hor_stride(frame, w);
    mpp_frame_set_ver_stride(frame, h);
    mpp_frame_set_fmt(frame, av_frame->format == AVPixelFormat::AV_PIX_FMT_RGB24
                                 ? MPP_FMT_RGB888
                                 : MPP_FMT_BGR888);
    mpp_frame_set_eos(frame, 0);
    mpp_frame_set_buffer(frame, mpp_buf);

    buffer = ColorChange(frame, RK_FORMAT_YCbCr_420_SP, device);
  }

  return buffer;
}

modelbox::Status FfmpegVideoEncoder::Encode(
    const std::shared_ptr<modelbox::Device> &device,
    const std::shared_ptr<AVFrame> &av_frame,
    std::vector<std::shared_ptr<AVPacket>> &av_packet_list) {
  std::lock_guard<std::mutex> lk(rk_enc_mtx_);
  auto av_buffer = FromAvFrame(device, av_frame);
  if (av_buffer == nullptr) {
    MBLOG_ERROR << "FromAvFrame fail";
    return {modelbox::STATUS_FAULT, "failed to FromAvFrame"};
  }

  auto *mpp_buf = (MppBuffer)(av_buffer->ConstData());
  MppFrame frame = nullptr;
  MppPacket packet = nullptr;

  auto ret = mpp_frame_init(&frame);
  if (ret != MPP_OK) {
    auto msg = std::string("mpp_frame_init failed: ") + std::to_string(ret);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  mpp_frame_set_width(frame, width_);
  mpp_frame_set_height(frame, height_);
  mpp_frame_set_hor_stride(frame, alignW_);
  mpp_frame_set_ver_stride(frame, alignH_);
  mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);
  mpp_frame_set_pts(frame, (RK_S64)(av_frame->pts));
  mpp_frame_set_eos(frame, 0);
  mpp_frame_set_buffer(frame, mpp_buf);

  Defer { mpp_frame_deinit(&frame); };

  ret = rk_api_->encode_put_frame(codec_ctx_, frame);
  if (ret != MPP_OK) {
    auto msg =
        std::string("mpp encode put frame failed: ") + std::to_string(ret);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  ret = rk_api_->encode_get_packet(codec_ctx_, &packet);
  if (ret != MPP_OK) {
    auto msg =
        std::string("mpp encode get packet failed: ") + std::to_string(ret);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  auto new_pkt = NewPacket(packet);
  if (new_pkt != nullptr) {
    av_packet_list.push_back(new_pkt);
  }

  return modelbox::STATUS_SUCCESS;
}