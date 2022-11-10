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

#include "local_camera_flowunit.h"

#include <securec.h>

#include <functional>
#include <nlohmann/json.hpp>

#include "modelbox/base/log.h"
#include "modelbox/base/utils.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"
#include "v4l2_camera.h"

#define RK_CAMERA_MAXRETRY 10

RKLocalCameraFlowUnit::RKLocalCameraFlowUnit() = default;
RKLocalCameraFlowUnit::~RKLocalCameraFlowUnit() = default;

modelbox::Status RKLocalCameraFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  camWidth_ = opts->GetInt32("cam_width", 640);
  camHeight_ = opts->GetInt32("cam_height", 480);
  camWidth_ = MPP_ALIGN(camWidth_, MPP_ALIGN_WIDTH);
  camHeight_ = MPP_ALIGN(camHeight_, MPP_ALIGN_HEIGHT);
  camera_id_ = (uint32_t)(opts->GetInt32("cam_id", 0));
  fps_ = (uint32_t)(opts->GetInt32("fps", 30));
  if (fps_ <= 0 || fps_ > 60) {
    fps_ = 30;
  }
  camera_bus_info_ = opts->GetString("bus_info", "");
  mirror_ = opts->GetBool("mirror", true);

  out_pix_fmt_str_ = opts->GetString("pix_fmt", modelbox::IMG_DEFAULT_FMT);
  MBLOG_INFO << "rockchip local-camera with " << out_pix_fmt_str_;

  out_pix_fmt_ = modelbox::GetRGAFormat(out_pix_fmt_str_);
  if (out_pix_fmt_ == RK_FORMAT_UNKNOWN) {
    MBLOG_ERROR << "Not support pix fmt " << out_pix_fmt_str_;
    return {modelbox::STATUS_BADCONF,
            "Not support pix fmt " + out_pix_fmt_str_};
  }

  return jpeg_dec_.Init();
}

modelbox::Status RKLocalCameraFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  std::string rk_source_url_ptr;
  auto input_meta = data_ctx->GetInputMeta(LOCAL_CAMERA_INPUT);
  if (input_meta != nullptr) {
    rk_source_url_ptr = *(
        std::static_pointer_cast<std::string>(input_meta->GetMeta(SOURCE_URL)));
  } else {
    try {
      auto buffer = data_ctx->Input(LOCAL_CAMERA_INPUT)->At(0);
      const char *inbuff_data = (const char *)buffer->ConstData();
      std::string input_cfg(inbuff_data, buffer->GetBytes());

      nlohmann::json json;
      json = nlohmann::json::parse(input_cfg);
      rk_source_url_ptr = json["url"].get<std::string>();
    } catch (const std::exception &e) {
      MBLOG_INFO << "no url, use default camera id or bus_info in graph";
    }
  }
  // check url is invalid or not
  if (!rk_source_url_ptr.empty()) {
    if ((rk_source_url_ptr.at(0) < '0' || rk_source_url_ptr.at(0) > '9') &&
        rk_source_url_ptr.substr(0, 4) != "usb-") {
      rk_source_url_ptr = "";
    }
  }

  if (rk_source_url_ptr.empty()) {
    if (camera_bus_info_.empty() || camera_bus_info_.substr(0, 4) != "usb-") {
      rk_source_url_ptr = std::to_string(camera_id_);
    } else {
      rk_source_url_ptr = camera_bus_info_;
    }
  }

  auto camhdl_ptr = std::make_shared<V4L2Camera>();
  bool prefer_rgb =
      (RK_FORMAT_RGB_888 == out_pix_fmt_ || RK_FORMAT_BGR_888 == out_pix_fmt_);
  auto ret = camhdl_ptr->Init(rk_source_url_ptr, camWidth_, camHeight_, fps_,
                              prefer_rgb);
  if (ret != modelbox::STATUS_SUCCESS) {
    auto msg = "camera url:" + rk_source_url_ptr +
               " init fail reason: " + ret.Errormsg();
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  auto frameindex_ptr = std::make_shared<int64_t>();
  *(frameindex_ptr.get()) = 0;
  auto retry_ptr = std::make_shared<int32_t>();
  *(retry_ptr.get()) = 0;

  data_ctx->SetPrivate(FRAME_INDEX_CTX, frameindex_ptr);
  data_ctx->SetPrivate(LOCAL_CAMERA_CTX, camhdl_ptr);
  data_ctx->SetPrivate(RETRY_COUNT_CTX, retry_ptr);
  MBLOG_INFO << "rknpu open local camera url = " << rk_source_url_ptr
             << " (w,h,fmt)=" << camhdl_ptr->GetWidth() << ","
             << camhdl_ptr->GetHeight() << "," << camhdl_ptr->GetFmt();

  return modelbox::STATUS_SUCCESS;
};

modelbox::Status RKLocalCameraFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  MBLOG_DEBUG << "rknpu local camera data post.";
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status RKLocalCameraFlowUnit::Close() {
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status RKLocalCameraFlowUnit::BuildOutput(
    const std::shared_ptr<modelbox::DataContext> &data_ctx,
    std::shared_ptr<modelbox::Buffer> &img_buf, MppFrame &frame,
    std::shared_ptr<int64_t> &frame_index) {
  auto output_bufs = data_ctx->Output(FRAME_INFO_OUTPUT);
  std::shared_ptr<modelbox::Buffer> buffer = nullptr;
  if (img_buf != nullptr &&
      out_pix_fmt_ == modelbox::GetRGAFormat(mpp_frame_get_fmt(frame))) {
    buffer = img_buf;
    auto w = (int32_t)mpp_frame_get_width(frame);
    auto h = (int32_t)mpp_frame_get_height(frame);
    auto ws = (int32_t)mpp_frame_get_hor_stride(frame);
    auto hs = MPP_ALIGN(h, MPP_ALIGN_HEIGHT);  // frame allign too large
    buffer->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
    buffer->Set("width", w);
    buffer->Set("height", h);
    int32_t channel = 3;
    if (RK_FORMAT_BGR_888 == out_pix_fmt_ ||
        RK_FORMAT_RGB_888 == out_pix_fmt_) {
      buffer->Set("width_stride", ws * 3);
    } else {
      buffer->Set("width_stride", ws);
      h = h * 3 / 2;
      hs = MPP_ALIGN(h, MPP_ALIGN_HEIGHT);
      channel = 1;
    }

    buffer->Set("channel", channel);
    buffer->Set("shape",
                std::vector<size_t>{(size_t)h, (size_t)w, (size_t)channel});
    buffer->Set("height_stride", hs);
    buffer->Set("layout", std::string("hwc"));
  } else {
    buffer = ColorChange(frame, out_pix_fmt_, GetBindDevice());
  }

  if (buffer != nullptr) {
    // update rk buffer
    buffer->Set("index", (*frame_index)++);
    buffer->Set("eos", false);
    buffer->Set("pix_fmt", out_pix_fmt_str_);

    std::shared_ptr<modelbox::Buffer> flip_buf = buffer;
    if (mirror_) {
      flip_buf = MirrorImg(buffer, out_pix_fmt_);
    }

    output_bufs->PushBack(flip_buf);
  }

  return modelbox::STATUS_CONTINUE;
}

MppFrame RKLocalCameraFlowUnit::ProcessYVY2(
    const uint8_t *buf, size_t size, size_t w, size_t h,
    std::shared_ptr<modelbox::Buffer> &img_buf) {
  MppFrame frame = nullptr;

  auto yuy2_buf = std::make_shared<modelbox::Buffer>(GetBindDevice());
  yuy2_buf->Build(size);
  auto *mpp_cam_buf = (MppBuffer)(yuy2_buf->MutableData());
  auto *cpu_cam_buf = (uint8_t *)mpp_buffer_get_ptr(mpp_cam_buf);
  auto yuy2_size = w * h;
  // yuy2 -- yuv422sp
  for (size_t i = 0; i < yuy2_size; i++) {
    cpu_cam_buf[i] = buf[i << 1];
    cpu_cam_buf[i + yuy2_size] = buf[(i << 1) + 1];
  }

  mpp_frame_init(&frame);
  mpp_frame_set_width(frame, w);
  mpp_frame_set_height(frame, h);
  mpp_frame_set_hor_stride(frame, w);
  mpp_frame_set_ver_stride(frame, h);
  mpp_frame_set_fmt(frame, MPP_FMT_YUV422SP);
  mpp_frame_set_eos(frame, 0);
  mpp_frame_set_buffer(frame, mpp_cam_buf);
  img_buf = yuy2_buf;

  return frame;
}

MppFrame RKLocalCameraFlowUnit::ProcessJpg(
    const uint8_t *buf, size_t size, size_t w, size_t h,
    std::shared_ptr<modelbox::Buffer> &img_buf) {
  // here make sure jpg_dec is locked, mpp jpeg dec is not thread-safe , only 1
  // jpg dec jpg dec onebyone may even faster
  std::lock_guard<std::mutex> lock(jpgdec_mtx_);
  // camera now in mjpg mode
  auto width = (int)w;
  auto height = (int)h;
  auto *frame = jpeg_dec_.Decode((void *)buf, size, width, height);
  if (frame == nullptr) {
    MBLOG_WARN << "local camera jpg decoder error";
  }

  return frame;
}

MppFrame RKLocalCameraFlowUnit::ProcessNV12(
    const uint8_t *buf, size_t size, size_t w, size_t h,
    std::shared_ptr<modelbox::Buffer> &img_buf) {
  MppFrame frame = nullptr;

  auto nv12_buf = std::make_shared<modelbox::Buffer>(GetBindDevice());
  nv12_buf->Build(size);
  auto *mpp_cam_buf = (MppBuffer)(nv12_buf->MutableData());
  auto *cpu_cam_buf = (uint8_t *)mpp_buffer_get_ptr(mpp_cam_buf);
  auto ret = memcpy_s(cpu_cam_buf, size, buf, size);
  if (ret != 0) {
    MBLOG_ERROR << "process nv12 memcpy fail";
    return nullptr;
  }

  mpp_frame_init(&frame);
  mpp_frame_set_width(frame, w);
  mpp_frame_set_height(frame, h);
  mpp_frame_set_hor_stride(frame, w);
  mpp_frame_set_ver_stride(frame, h);
  mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);
  mpp_frame_set_eos(frame, 0);
  mpp_frame_set_buffer(frame, mpp_cam_buf);
  img_buf = nv12_buf;

  return frame;
}

MppFrame RKLocalCameraFlowUnit::ProcessRGB24(
    const uint8_t *buf, size_t size, size_t w, size_t h,
    std::shared_ptr<modelbox::Buffer> &img_buf) {
  MppFrame frame = nullptr;

  auto rgb24_buf = std::make_shared<modelbox::Buffer>(GetBindDevice());
  rgb24_buf->Build(size);
  auto *mpp_cam_buf = (MppBuffer)(rgb24_buf->MutableData());
  auto *cpu_cam_buf = (uint8_t *)mpp_buffer_get_ptr(mpp_cam_buf);
  auto ret = memcpy_s(cpu_cam_buf, size, buf, size);
  if (ret != 0) {
    MBLOG_ERROR << "process rgb memcpy fail";
    return nullptr;
  }

  mpp_frame_init(&frame);
  mpp_frame_set_width(frame, w);
  mpp_frame_set_height(frame, h);
  mpp_frame_set_hor_stride(frame, w);
  mpp_frame_set_ver_stride(frame, h);
  mpp_frame_set_fmt(frame, MPP_FMT_RGB888);
  mpp_frame_set_eos(frame, 0);
  mpp_frame_set_buffer(frame, mpp_cam_buf);
  img_buf = rgb24_buf;

  return frame;
}

modelbox::Status RKLocalCameraFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto camhdl_ptr = std::static_pointer_cast<V4L2Camera>(
      data_ctx->GetPrivate(LOCAL_CAMERA_CTX));
  auto frame_index =
      std::static_pointer_cast<int64_t>(data_ctx->GetPrivate(FRAME_INDEX_CTX));
  auto retry_count =
      std::static_pointer_cast<int32_t>(data_ctx->GetPrivate(RETRY_COUNT_CTX));
  if (camhdl_ptr == nullptr || frame_index == nullptr ||
      retry_count == nullptr) {
    MBLOG_ERROR << "localcamera is not init";
    return {modelbox::STATUS_FAULT, "localcamera is not init"};
  }

  auto ret = modelbox::STATUS_CONTINUE;

  Defer {
    if (ret == modelbox::STATUS_CONTINUE) {
      auto event = std::make_shared<modelbox::FlowUnitEvent>();
      data_ctx->SendEvent(event);
    }
  };

  using FuncType =
      std::function<MppFrame(RKLocalCameraFlowUnit *, uint8_t *, size_t, size_t,
                             size_t, std::shared_ptr<modelbox::Buffer> &)>;
  static const std::map<uint32_t, FuncType> process_funcs = {
      {V4L2_PIX_FMT_MJPEG, &RKLocalCameraFlowUnit::ProcessJpg},
      {V4L2_PIX_FMT_NV12, &RKLocalCameraFlowUnit::ProcessNV12},
      {V4L2_PIX_FMT_RGB24, &RKLocalCameraFlowUnit::ProcessRGB24},
      {V4L2_PIX_FMT_YUYV, &RKLocalCameraFlowUnit::ProcessYVY2},
  };

  MppFrame frame = nullptr;
  std::shared_ptr<modelbox::Buffer> img_buf = nullptr;
  auto cam_buf = camhdl_ptr->GetFrame();
  auto iter = process_funcs.find(camhdl_ptr->GetFmt());
  if (cam_buf != nullptr && iter != process_funcs.end()) {
    frame =
        iter->second(this, (uint8_t *)(cam_buf->start), cam_buf->length,
                     camhdl_ptr->GetWidth(), camhdl_ptr->GetHeight(), img_buf);
  }
  if (frame == nullptr) {
    // log has been put in ProcessYVY2 or ProcessJpg
    if ((*retry_count)++ > RK_CAMERA_MAXRETRY) {
      MBLOG_ERROR << "localcamera get buffer fail";
      return {modelbox::STATUS_FAULT, "localcamera get buffer fail"};
    }

    MBLOG_WARN << "local camera get null buffer";
    return modelbox::STATUS_CONTINUE;
  }

  *retry_count = 0;
  return BuildOutput(data_ctx, img_buf, frame, frame_index);
}

MODELBOX_FLOWUNIT(RKLocalCameraFlowUnit, rk_cam_desc) {
  rk_cam_desc.SetFlowUnitName(FLOWUNIT_NAME);
  rk_cam_desc.SetFlowUnitGroupType("Video");
  rk_cam_desc.AddFlowUnitInput({LOCAL_CAMERA_INPUT, "cpu"});
  rk_cam_desc.AddFlowUnitOutput({FRAME_INFO_OUTPUT, modelbox::DEVICE_TYPE});
  rk_cam_desc.SetFlowType(modelbox::STREAM);
  rk_cam_desc.SetInputContiguous(false);
  rk_cam_desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "cam_width", "int", false, "0", "the camera width"));
  rk_cam_desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "cam_height", "int", false, "0", "the camera height"));
  rk_cam_desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("cam_id", "int", false, "-1", "the camera id"));
  rk_cam_desc.AddFlowUnitOption(
      modelbox::FlowUnitOption("fps", "int", false, "30", "the camera fps"));
  rk_cam_desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "bus_info", "string", false, "",
      "v4l2 camera bus_info, use v4l2-ctl --list-devices"));
  rk_cam_desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "pix_fmt", "string", false, modelbox::IMG_DEFAULT_FMT, "the pix format"));
  rk_cam_desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "mirror", "bool", false, "true", "camera mirror"));
}

MODELBOX_DRIVER_FLOWUNIT(rk_cam_desc) {
  rk_cam_desc.Desc.SetName(FLOWUNIT_NAME);
  rk_cam_desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  rk_cam_desc.Desc.SetType(modelbox::DEVICE_TYPE);
  rk_cam_desc.Desc.SetDescription(FLOWUNIT_DESC);
  rk_cam_desc.Desc.SetVersion(MODELBOX_VERSION_STR_MACRO);
}
