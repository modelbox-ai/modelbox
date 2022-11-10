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

#ifndef _V4L2_CAMERA_H_
#define _V4L2_CAMERA_H_

#include <linux/videodev2.h>
#include <modelbox/base/status.h>

#include <string>

#define RK_CAMERA_BUFCNT 4

typedef struct CamFrame_t {
  void *start;
  size_t length;
} CamFrame;

class V4L2Camera {
 public:
  V4L2Camera();
  ~V4L2Camera();

  modelbox::Status Init(const std::string &cam_url, uint32_t cam_width,
                        uint32_t cam_height, uint32_t fps, bool prefer_rgb);
  std::shared_ptr<CamFrame> GetFrame();
  inline uint32_t GetWidth() { return width_; }
  inline uint32_t GetHeight() { return height_; }
  inline uint32_t GetFmt() { return cam_fmt_; }

 private:
  modelbox::Status CamIoCtl(int32_t fd, int32_t req, void *arg);
  modelbox::Status SetFmt(uint32_t cam_width, uint32_t cam_height,
                          bool prefer_rgb);
  modelbox::Status SetFps(uint32_t fps);
  modelbox::Status RequestBuf();
  modelbox::Status MapMemory();
  modelbox::Status QBufAndRun();
  void PutFrame(uint32_t idx, CamFrame *p);
  int32_t GetCamfd(int32_t id, const std::string &bus_info);
  bool IsCamera(int32_t fd, uint32_t capabilities, const char *cam_name);

 private:
  int32_t fd_{-1};
  uint32_t cam_fmt_{V4L2_PIX_FMT_MJPEG};
  enum v4l2_buf_type type_ { V4L2_BUF_TYPE_VIDEO_CAPTURE };
  uint32_t width_{0};
  uint32_t height_{0};
  CamFrame fbuf_[RK_CAMERA_BUFCNT];         // frame buffers
  CamFrame shared_fbuf_[RK_CAMERA_BUFCNT];  // frame buffers
};

#endif