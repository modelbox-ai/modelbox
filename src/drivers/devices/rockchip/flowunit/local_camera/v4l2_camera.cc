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

#include "v4l2_camera.h"

#include <fcntl.h>
#include <glob.h>
#include <securec.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "modelbox/base/log.h"

constexpr const char *CAM_DEV = "/dev/video";
#define FMT_NUM_PLANES 1
#define SKIP_COUNT 10

V4L2Camera::V4L2Camera() = default;

V4L2Camera::~V4L2Camera() {
  if (fd_ < 0) {
    return;
  }

  // Stop capturing
  CamIoCtl(fd_, VIDIOC_STREAMOFF, &type_);

  // un-mmap buffers
  for (size_t i = 0; i < RK_CAMERA_BUFCNT; i++) {
    struct v4l2_buffer buf = {0};
    buf.type = type_;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    CamIoCtl(fd_, VIDIOC_QUERYBUF, &buf);
    // no mpp_buffer , not need put
    munmap(fbuf_[i].start, buf.length);
  }

  // Close v4l2 device
  close(fd_);
  fd_ = -1;
}

modelbox::Status V4L2Camera::CamIoCtl(int32_t fd, int32_t req, void *arg) {
  int32_t ret;

  while ((ret = ioctl(fd, req, arg))) {
    if (ret == -1 && (EINTR != errno && EAGAIN != errno)) {
      break;
    }
    // 10 milliseconds
    usleep(1000 * 10);
  }

  if (ret == 0) {
    return modelbox::STATUS_SUCCESS;
  }

  auto msg = std::string("ioctl fail errno: ") + modelbox::StrError(errno);
  MBLOG_ERROR << msg;
  return {modelbox::STATUS_FAULT, msg};
}

int32_t V4L2Camera::GetCamfd(int32_t id, const std::string &bus_info) {
  int32_t cam_index = 0;
  int32_t fd = -1;
  struct v4l2_capability cap;
  glob_t glob_result;
  struct stat buffer;
  bool bfound = false;

  auto ret = glob("/dev/video*", GLOB_TILDE, nullptr, &glob_result);
  if (ret != 0) {
    return -1;
  }

  // do not add any return , globfree(&glob_result);
  for (unsigned int i = 0; i < glob_result.gl_pathc; i++) {
    if (stat(glob_result.gl_pathv[i], &buffer) == -1) {
      continue;
    }

    if (S_ISDIR(buffer.st_mode) != 0) {
      continue;
    }

    if (fd >= 0) {
      close(fd);
      fd = -1;
    }

    fd = open(glob_result.gl_pathv[i], O_RDWR, 0);
    if (fd < 0) {
      MBLOG_DEBUG << "Cannot open device:" << glob_result.gl_pathv[i];
      continue;
    }

    Defer { globfree(&glob_result); };
	
    // detect it is a camera device
    if (modelbox::STATUS_SUCCESS != CamIoCtl(fd, VIDIOC_QUERYCAP, &cap)) {
      MBLOG_DEBUG << "Not v4l2 device:" << glob_result.gl_pathv[i];
      continue;
    }

    if (!IsCamera(fd, cap.capabilities, glob_result.gl_pathv[i])) {
      continue;
    }

    cam_index++;
    if (bus_info.empty()) {
      if (cam_index == (id + 1)) {
        // find id camera, return the fd
        bfound = true;
        break;
      }
    } else {
      // if bus_info, find the right v4l2 device
      if (bus_info == (const char *)(cap.bus_info)) {
        bfound = true;
        break;
      }
    }
  }

  if (!bfound && fd >= 0) {
    close(fd);
    fd = -1;
  }

  return fd;
}

bool V4L2Camera::IsCamera(int32_t fd, uint32_t capabilities,
                          const char *cam_name) {
  if (!(capabilities & V4L2_CAP_VIDEO_CAPTURE) &&
      !(capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE)) {
    MBLOG_DEBUG << "Camera Capture not supported for" << cam_name;
    return false;
  }

  if (!(capabilities & V4L2_CAP_STREAMING)) {
    MBLOG_DEBUG << "Camera Streaming IO Not Supported for " << cam_name;
    return false;
  }

  if (capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) {
    type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  } else {
    type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  }

  struct v4l2_format vfmt = {0};
  vfmt.type = type_;
  if (modelbox::STATUS_SUCCESS != CamIoCtl(fd, VIDIOC_G_FMT, &vfmt)) {
    MBLOG_DEBUG << "Camera VIDIOC_G_FMT fail for " << cam_name;
    return false;
  }

  if (vfmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG &&
      vfmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
    MBLOG_DEBUG << "Camera fmt:" << vfmt.fmt.pix.pixelformat
                << " not support for " << cam_name;
    return false;
  }

  return true;
}

modelbox::Status V4L2Camera::SetFmt(uint32_t cam_width, uint32_t cam_height,
                                    bool prefer_rgb) {
  struct v4l2_format vfmt = {0};
  vfmt.type = type_;
  vfmt.fmt.pix.width = cam_width;
  vfmt.fmt.pix.height = cam_height;

  std::vector<uint32_t> try_fmts = {V4L2_PIX_FMT_MJPEG, V4L2_PIX_FMT_NV12,
                                    V4L2_PIX_FMT_NV12, V4L2_PIX_FMT_YUYV};
  if (prefer_rgb) {
    try_fmts[1] = V4L2_PIX_FMT_RGB24;
  }

  // 有优先级顺序，一定要优先使用MJPEG
  uint32_t i = 0;
  for (i = 0; i < try_fmts.size(); i++) {
    vfmt.fmt.pix.pixelformat = try_fmts[i];
    if (modelbox::STATUS_SUCCESS == CamIoCtl(fd_, VIDIOC_S_FMT, &vfmt)) {
      break;
    }
  }

  if (modelbox::STATUS_SUCCESS != CamIoCtl(fd_, VIDIOC_G_FMT, &vfmt)) {
    MBLOG_ERROR << "VIDIOC_G_FMT fail";
    return {modelbox::STATUS_FAULT, "VIDIOC_G_FMT fail"};
  }

  // 设置失败，使用默认值
  if (i == try_fmts.size()) {
    MBLOG_WARN << "VIDIOC_S_FMT fail, use the default value";
    if (CamIoCtl(fd_, VIDIOC_S_FMT, &vfmt) != modelbox::STATUS_SUCCESS) {
      MBLOG_ERROR << "failed to cam io ctl";
      return {modelbox::STATUS_FAULT, "failed to cam io ctl"};
    }
  }

  cam_fmt_ = vfmt.fmt.pix.pixelformat;
  width_ = vfmt.fmt.pix.width;
  height_ = vfmt.fmt.pix.height;

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status V4L2Camera::SetFps(uint32_t fps) {
  struct v4l2_streamparm setfps = {0};
  setfps.type = type_;
  setfps.parm.capture.timeperframe.numerator = 1;
  setfps.parm.capture.timeperframe.denominator = fps;
  if (modelbox::STATUS_SUCCESS != CamIoCtl(fd_, VIDIOC_S_PARM, &setfps)) {
    MBLOG_WARN << "VIDIOC_S_PARM set fps fail";
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status V4L2Camera::RequestBuf() {
  struct v4l2_requestbuffers req = {0};
  req.count = RK_CAMERA_BUFCNT;
  req.type = type_;
  req.memory = V4L2_MEMORY_MMAP;
  auto ret = CamIoCtl(fd_, VIDIOC_REQBUFS, &req);
  if (modelbox::STATUS_SUCCESS != ret) {
    MBLOG_ERROR << "Device does not support mmap";
    return {modelbox::STATUS_FAULT,
            "Device does not support mmap reason: " + ret.Errormsg()};
  }

  if (req.count != RK_CAMERA_BUFCNT) {
    MBLOG_ERROR << "Device buffer count mismatch";
    return {modelbox::STATUS_FAULT, "Device buffer count mismatch"};
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status V4L2Camera::MapMemory() {
  uint32_t buf_len = 0;
  uint32_t offset = 0;
  // mmap the v4l2 buf into userspace memory
  for (uint32_t i = 0; i < RK_CAMERA_BUFCNT; i++) {
    struct v4l2_buffer buf = {0};
    buf.type = type_;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    struct v4l2_plane planes[FMT_NUM_PLANES];
    buf.memory = V4L2_MEMORY_MMAP;
    if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == type_) {
      buf.m.planes = planes;
      buf.length = FMT_NUM_PLANES;
    }

    auto ret = CamIoCtl(fd_, VIDIOC_QUERYBUF, &buf);
    if (modelbox::STATUS_SUCCESS != ret) {
      MBLOG_ERROR << "VIDIOC_QUERYBUF fail";
      return {modelbox::STATUS_FAULT,
              "VIDIOC_QUERYBUF fail reason: " + ret.Errormsg()};
    }

    if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == buf.type) {
      buf_len = buf.m.planes[0].length;
      offset = buf.m.planes[0].m.mem_offset;
    } else {
      buf_len = buf.length;
      offset = buf.m.offset;
    }

    fbuf_[i].start =
        mmap(nullptr, buf_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, offset);
    shared_fbuf_[i].start = fbuf_[i].start;
    if (MAP_FAILED == fbuf_[i].start) {
      MBLOG_ERROR << "Failed to map device frame buffers";
      return {modelbox::STATUS_FAULT, "Failed to map device frame buffers"};
    }
    // do not map to mpp_buffer , dma_buffer seems fail in jpg_dec
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status V4L2Camera::QBufAndRun() {
  // qbuf into v4l2
  for (size_t i = 0; i < RK_CAMERA_BUFCNT; i++) {
    struct v4l2_plane planes[FMT_NUM_PLANES];
    struct v4l2_buffer buf = {0};
    buf.type = type_;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    buf.memory = V4L2_MEMORY_MMAP;

    if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == type_) {
      buf.m.planes = planes;
      buf.length = FMT_NUM_PLANES;
    }

    auto ret = CamIoCtl(fd_, VIDIOC_QBUF, &buf);
    if (modelbox::STATUS_SUCCESS != ret) {
      MBLOG_ERROR << "VIDIOC_QBUF fail at :" << i
                  << " reason: " << ret.Errormsg();
      return {modelbox::STATUS_FAULT,
              "VIDIOC_QBUF fail at :" + std::to_string(i) +
                  " reason: " + ret.Errormsg()};
    }
  }

  // Start capturing
  enum v4l2_buf_type type = type_;
  auto ret = CamIoCtl(fd_, VIDIOC_STREAMON, &type);
  if (modelbox::STATUS_SUCCESS != ret) {
    MBLOG_ERROR << "VIDIOC_STREAMON fail reason: " << ret.Errormsg();
    return {modelbox::STATUS_FAULT,
            "VIDIOC_STREAMON fail reason: " + ret.Errormsg()};
  }

  // skip some frames at start
  for (size_t i = 0; i < SKIP_COUNT; i++) {
    GetFrame();
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status V4L2Camera::Init(const std::string &cam_url,
                                  uint32_t cam_width, uint32_t cam_height,
                                  uint32_t fps, bool prefer_rgb) {
  int32_t id = -1;
  std::string cam_name = cam_url;
  try {
    // string -> integer
    id = std::stoi(cam_name);
    // set empty
    cam_name = "";
  } catch (const std::exception &e) {
    auto msg =
        "stoi exception v4l2 camera name: " + cam_name + " reason: " + e.what();
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  fd_ = GetCamfd(id, cam_name);
  if (fd_ < 0) {
    auto msg = "can not find v4l2 camera name: " + cam_name;
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  auto ret = SetFmt(cam_width, cam_height, prefer_rgb);
  if (modelbox::STATUS_SUCCESS != ret) {
    return {modelbox::STATUS_FAULT,
            "failed to SetFmt reason: " + ret.Errormsg()};
  }

  ret = SetFps(fps);
  if (modelbox::STATUS_SUCCESS != ret) {
    return {modelbox::STATUS_FAULT,
            "failed to SetFps reason: " + ret.Errormsg()};
  }

  ret = RequestBuf();
  if (modelbox::STATUS_SUCCESS != ret) {
    return {modelbox::STATUS_FAULT,
            "failed to RequestBuf reason: " + ret.Errormsg()};
  }

  ret = MapMemory();
  if (modelbox::STATUS_SUCCESS != ret) {
    return {modelbox::STATUS_FAULT,
            "failed to MapMemory reason: " + ret.Errormsg()};
  }

  ret = QBufAndRun();
  if (modelbox::STATUS_SUCCESS != ret) {
    return {modelbox::STATUS_FAULT,
            "failed to QBufAndRun reason: " + ret.Errormsg()};
  }

  return modelbox::STATUS_SUCCESS;
}

std::shared_ptr<CamFrame> V4L2Camera::GetFrame() {
  struct v4l2_buffer buf = {0};
  buf.type = type_;
  buf.memory = V4L2_MEMORY_MMAP;

  struct v4l2_plane planes[FMT_NUM_PLANES];
  if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == type_) {
    buf.m.planes = planes;
    buf.length = FMT_NUM_PLANES;
  }

  if (modelbox::STATUS_SUCCESS != CamIoCtl(fd_, VIDIOC_DQBUF, &buf)) {
    MBLOG_ERROR << "GetFrame VIDIOC_DQBUF fail";
    return nullptr;
  }

  if (buf.index >= RK_CAMERA_BUFCNT) {
    MBLOG_ERROR << "GetFrame buffer index out of bounds";
    return nullptr;
  }

  if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == type_) {
    buf.bytesused = buf.m.planes[0].bytesused;
  }

  shared_fbuf_[buf.index].length = buf.bytesused;
  shared_fbuf_[buf.index].start = fbuf_[buf.index].start;
  std::shared_ptr<CamFrame> ret(
      &shared_fbuf_[buf.index],
      std::bind(&V4L2Camera::PutFrame, this, buf.index, std::placeholders::_1));

  return ret;
}

// It's OK to capture into this framebuffer now
void V4L2Camera::PutFrame(uint32_t idx, CamFrame *p) {
  // do not delete p, it's class local var
  struct v4l2_buffer buf = {0};
  buf.type = type_;
  buf.memory = V4L2_MEMORY_MMAP;
  buf.index = idx;

  struct v4l2_plane planes[FMT_NUM_PLANES];
  if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == type_) {
    buf.m.planes = planes;
    buf.length = FMT_NUM_PLANES;
  }

  auto ret = CamIoCtl(fd_, VIDIOC_QBUF, &buf);
  if (modelbox::STATUS_SUCCESS != ret) {
    MBLOG_ERROR << "PutFrame VIDIOC_QBUF fail reason: " << ret.Errormsg();
  }
}