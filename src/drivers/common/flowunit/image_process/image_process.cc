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

#include "image_process.h"

#include <functional>
#include <map>
#include <set>

namespace imageprocess {
int32_t align_up(int32_t num, int32_t align) {
  if (align == 0 || num == 0) {
    return 0;
  }

  return ((num - 1) / align + 1) * align;
}

static const std::set<std::string> pix_fmt_set = {"rgb", "bgr", "nv12", "nv21"};

static size_t RGBBytesCalc(size_t pix_num) { return pix_num * 3; }

static size_t NVBytesCalc(size_t pix_num) { return pix_num * 3 / 2; }

static const std::map<std::string, std::function<size_t(size_t)>>
    img_bytes_calc_map = {{"rgb", RGBBytesCalc},
                          {"bgr", RGBBytesCalc},
                          {"nv12", NVBytesCalc},
                          {"nv21", NVBytesCalc}};

modelbox::Status GetImageBytes(const std::string &pix_fmt, size_t pix_num,
                               size_t &img_bytes) {
  auto item = img_bytes_calc_map.find(pix_fmt);
  if (item == img_bytes_calc_map.end()) {
    return {modelbox::STATUS_NOTSUPPORT,
            "pix_fmt " + pix_fmt + " is not support"};
  }

  img_bytes = item->second(pix_num);
  return modelbox::STATUS_OK;
}

modelbox::Status GetImageBytes(const std::string &pix_fmt, int32_t width,
                               int32_t height, size_t &img_bytes) {
  return GetImageBytes(pix_fmt, (size_t)width * height, img_bytes);
}

static int32_t RGBWidthStrideCalc(int32_t width) { return width * 3; }

static int32_t NVWidthStrideCalc(int32_t width) { return width; }

static const std::map<std::string, std::function<int32_t(int32_t)>>
    img_width_stride_calc_map = {{"rgb", RGBWidthStrideCalc},
                                 {"bgr", RGBWidthStrideCalc},
                                 {"nv12", NVWidthStrideCalc},
                                 {"nv21", NVWidthStrideCalc}};

modelbox::Status GetWidthStride(const std::string &pix_fmt, int32_t width,
                                int32_t &width_stride) {
  auto item = img_width_stride_calc_map.find(pix_fmt);
  if (item == img_width_stride_calc_map.end()) {
    return {modelbox::STATUS_NOTSUPPORT,
            "pix_fmt " + pix_fmt + " is not support"};
  }

  width_stride = item->second(width);
  return modelbox::STATUS_OK;
}

static size_t RGBBytesCalcByStride(int32_t width_stride,
                                   int32_t height_stride) {
  return (size_t)width_stride * height_stride * 3;
}

static size_t NVBytesCalcByStride(int32_t width_stride, int32_t height_stride) {
  return (size_t)width_stride * height_stride * 3 / 2;
}

static const std::map<std::string, std::function<size_t(int32_t, int32_t)>>
    img_bytes_calc_map2 = {{"rgb", RGBBytesCalcByStride},
                           {"bgr", RGBBytesCalcByStride},
                           {"nv12", NVBytesCalcByStride},
                           {"nv21", NVBytesCalcByStride}};

modelbox::Status GetImageBytesByStride(const std::string &pix_fmt,
                                       int32_t width_stride,
                                       int32_t height_stride,
                                       size_t &img_bytes) {
  auto item = img_bytes_calc_map2.find(pix_fmt);
  if (item == img_bytes_calc_map2.end()) {
    return {modelbox::STATUS_NOTSUPPORT,
            "pix_fmt " + pix_fmt + " is not support"};
  }

  img_bytes = item->second(width_stride, height_stride);
  return modelbox::STATUS_OK;
}

modelbox::Status GetImgParam(const std::shared_ptr<modelbox::Buffer> &img,
                             std::string &pix_fmt, int32_t &img_width,
                             int32_t &img_height, int32_t &img_width_stride,
                             int32_t &img_height_stride) {
  auto b_ret = img->Get("pix_fmt", pix_fmt);
  if (!b_ret) {
    return {modelbox::STATUS_INVALID, "pix_fmt not in input image meta"};
  }

  b_ret = img->Get("width", img_width);
  if (!b_ret) {
    return {modelbox::STATUS_INVALID, "width not in input image meta"};
  }

  b_ret = img->Get("height", img_height);
  if (!b_ret) {
    return {modelbox::STATUS_INVALID, "height not in input image meta"};
  }

  b_ret = img->Get("width_stride", img_width_stride);
  if (!b_ret) {
    return {modelbox::STATUS_INVALID, "width stride not in input image meta"};
  }

  b_ret = img->Get("height_stride", img_height_stride);
  if (!b_ret) {
    return {modelbox::STATUS_INVALID, "height stride not in input image meta"};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status CheckImageStride(const std::string &pix_fmt,
                                  int32_t img_width_stride,
                                  int32_t expect_w_align,
                                  int32_t img_height_stride,
                                  int32_t expect_h_align, size_t img_size) {
  if (expect_w_align == 0 || expect_h_align == 0) {
    return {modelbox::STATUS_NOTSUPPORT, "divisor is zero"};
  }
  if (img_width_stride % expect_w_align != 0) {
    return {modelbox::STATUS_NOTSUPPORT,
            "img_width_stride must align to " + std::to_string(expect_w_align)};
  }

  if (img_height_stride % expect_h_align != 0) {
    return {modelbox::STATUS_NOTSUPPORT, "img_height_stride must align to " +
                                             std::to_string(expect_h_align)};
  }

  size_t expect_size = 0;
  auto ret = GetImageBytesByStride(pix_fmt, img_width_stride, img_height_stride,
                                   expect_size);
  if (!ret) {
    return ret;
  }

  if (img_size != expect_size) {
    return {modelbox::STATUS_INVALID,
            "img_size[ " + std::to_string(img_size) + " ] not right, pix_fmt[" +
                pix_fmt + "], width_stride[" +
                std::to_string(img_width_stride) + "], height_stride[" +
                std::to_string(img_height_stride) + "]"};
  }

  return modelbox::STATUS_OK;
}

#ifdef ACL_ENABLE

modelbox::Status InitDvppChannel(
    std::shared_ptr<acldvppChannelDesc> &chan_desc) {
  auto chan_desc_ptr = acldvppCreateChannelDesc();
  if (chan_desc_ptr == nullptr) {
    return {modelbox::STATUS_FAULT, "acldvppCreateChannelDesc return null"};
  }

  auto acl_ret = acldvppCreateChannel(chan_desc_ptr);
  if (acl_ret != ACL_SUCCESS) {
    acldvppDestroyChannelDesc(chan_desc_ptr);
    return {modelbox::STATUS_FAULT,
            "acldvppCreateChannel failed, ret " + std::to_string(acl_ret)};
  }

  chan_desc.reset(chan_desc_ptr, [](acldvppChannelDesc *ptr) {
    acldvppDestroyChannel(ptr);
    acldvppDestroyChannelDesc(ptr);
  });

  return modelbox::STATUS_SUCCESS;
}

static std::map<std::string, acldvppPixelFormat> acl_fmt_trans_map = {
    {"nv12", PIXEL_FORMAT_YUV_SEMIPLANAR_420},
    {"nv21", PIXEL_FORMAT_YVU_SEMIPLANAR_420},
    {"rgb", PIXEL_FORMAT_RGB_888},
    {"bgr", PIXEL_FORMAT_BGR_888},
};

std::shared_ptr<acldvppPicDesc> CreateImgDesc(size_t img_size,
                                              const std::string &pix_fmt,
                                              const ImageShape &shape,
                                              ImgDescDestroyFlag flag) {
  void *buffer = nullptr;
  auto acl_ret = acldvppMalloc(&buffer, img_size);
  if (acl_ret != ACL_SUCCESS) {
    modelbox::StatusError = {
        modelbox::STATUS_FAULT,
        "acldvppMalloc failed, code " + std::to_string(acl_ret)};
    return nullptr;
  }

  auto img_desc = CreateImgDesc(img_size, buffer, pix_fmt, shape, flag);
  if (img_desc == nullptr) {
    acldvppFree(buffer);
    return nullptr;
  }

  modelbox::StatusError = modelbox::STATUS_OK;
  return img_desc;
}

std::shared_ptr<acldvppPicDesc> CreateImgDesc(size_t img_size, void *img_buffer,
                                              const std::string &pix_fmt,
                                              const ImageShape &shape,
                                              ImgDescDestroyFlag flag) {
  auto ret =
      CheckImageStride(pix_fmt, shape.width_stride, ASCEND_WIDTH_ALIGN,
                       shape.height_stride, ASCEND_HEIGHT_ALIGN, img_size);
  if (!ret) {
    modelbox::StatusError = ret;
    return nullptr;
  }

  auto img_desc_ptr = acldvppCreatePicDesc();
  if (img_desc_ptr == nullptr) {
    modelbox::StatusError = {modelbox::STATUS_FAULT,
                             "acldvppCreatePicDesc return null"};
    return nullptr;
  }

  auto format_item = acl_fmt_trans_map.find(pix_fmt);
  if (format_item == acl_fmt_trans_map.end()) {
    acldvppDestroyPicDesc(img_desc_ptr);
    modelbox::StatusError = {modelbox::STATUS_NOTSUPPORT,
                             "pix_fmt " + pix_fmt + " is not support"};
    return nullptr;
  }

  acldvppSetPicDescSize(img_desc_ptr, img_size);
  if (img_buffer != nullptr) {
    acldvppSetPicDescData(img_desc_ptr, img_buffer);
  }
  acldvppSetPicDescFormat(img_desc_ptr, format_item->second);
  acldvppSetPicDescWidth(img_desc_ptr, shape.width);
  acldvppSetPicDescHeight(img_desc_ptr, shape.height);
  acldvppSetPicDescWidthStride(img_desc_ptr, shape.width_stride);
  acldvppSetPicDescHeightStride(img_desc_ptr, shape.height_stride);

  std::shared_ptr<acldvppPicDesc> img_desc(
      img_desc_ptr, [flag](acldvppPicDesc *ptr) {
        if (flag == ImgDescDestroyFlag::NONE) {
          return;
        }

        auto data = acldvppGetPicDescData(ptr);
        if (data != nullptr && flag != ImgDescDestroyFlag::DESC_ONLY) {
          acldvppFree(data);
        }

        acldvppDestroyPicDesc(ptr);
      });

  modelbox::StatusError = modelbox::STATUS_OK;
  return img_desc;
}

modelbox::Status FillImgDescData(
    std::shared_ptr<acldvppPicDesc> &img_desc,
    std::shared_ptr<modelbox::Buffer> &image_buffer, aclrtStream stream) {
  auto acl_ret = aclrtSynchronizeStream(stream);
  if (acl_ret != ACL_SUCCESS) {
    MBLOG_ERROR << "aclrtSynchronizeStream failed, err " << acl_ret;
    return modelbox::STATUS_FAULT;
  }

  auto input_buf = acldvppGetPicDescData(img_desc.get());
  if (input_buf == nullptr) {
    return {modelbox::STATUS_FAULT, "acldvppGetPicDescData failed"};
  }

  acl_ret = aclrtMemcpy(input_buf, image_buffer->GetBytes(),
                        image_buffer->ConstData(), image_buffer->GetBytes(),
                        aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE);
  if (acl_ret != ACL_SUCCESS) {
    std::string err_msg =
        "aclrtMemcpyAsync failed, dest_ptr:" +
        std::to_string((uintptr_t)input_buf) +
        ",dest_size:" + std::to_string(image_buffer->GetBytes()) +
        ",src_ptr:" + std::to_string((uintptr_t)image_buffer->ConstData()) +
        ",src_size:" + std::to_string(image_buffer->GetBytes()) +
        ",type:device_to_device";
    return {modelbox::STATUS_FAULT, err_msg};
  }

  return modelbox::STATUS_OK;
}

modelbox::Status SetOutImgMeta(std::shared_ptr<modelbox::Buffer> &out_image,
                               const std::string &out_pix_fmt,
                               std::shared_ptr<acldvppPicDesc> &out_img_desc) {
  auto img_desc_ptr = out_img_desc.get();
  if (img_desc_ptr == nullptr) {
    return {modelbox::STATUS_FAULT, "out_img_desc is null"};
  }

  int32_t width = acldvppGetPicDescWidth(img_desc_ptr);
  int32_t height = acldvppGetPicDescHeight(img_desc_ptr);
  int32_t width_stride = acldvppGetPicDescWidthStride(img_desc_ptr);
  int32_t height_stride = acldvppGetPicDescHeightStride(img_desc_ptr);
  out_image->Set("width", width);
  out_image->Set("height", height);
  out_image->Set("width_stride", width_stride);
  out_image->Set("height_stride", height_stride);
  out_image->Set("channel", (int32_t)1);
  out_image->Set("pix_fmt", out_pix_fmt);
  out_image->Set("layout", std::string("hwc"));
  out_image->Set("shape", std::vector<size_t>{(size_t)height_stride * 3 / 2,
                                              (size_t)width_stride, 1});
  out_image->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
  return modelbox::STATUS_OK;
}

class DvppChanMgr : public std::enable_shared_from_this<DvppChanMgr> {
 public:
  std::shared_ptr<acldvppChannelDesc> Get(int32_t device_id) {
    std::shared_ptr<acldvppChannelDesc> chan;
    std::weak_ptr<DvppChanMgr> mgr_ref = shared_from_this();
    auto free_func = [mgr_ref, device_id](acldvppChannelDesc *ptr) {
      auto mgr = mgr_ref.lock();
      if (mgr == nullptr) {
        acldvppDestroyChannel(ptr);
        acldvppDestroyChannelDesc(ptr);
        return;
      }

      mgr->Put(ptr, device_id);
    };

    {
      std::lock_guard<std::mutex> lock(chan_list_lock_);
      auto &chan_list = device_chan_list_[device_id];
      if (!chan_list.empty()) {
        auto ch_ptr = chan_list.front();
        chan_list.pop_front();
        chan.reset(ch_ptr, free_func);
        return chan;
      }
    }

    auto ch_ptr = acldvppCreateChannelDesc();
    if (ch_ptr == nullptr) {
      MBLOG_ERROR << "acldvppCreateChannelDesc return null";
      return nullptr;
    }

    auto acl_ret = acldvppCreateChannel(ch_ptr);
    if (acl_ret != ACL_SUCCESS) {
      acldvppDestroyChannelDesc(ch_ptr);
      MBLOG_ERROR << "acldvppCreateChannel failed, acl ret " << acl_ret;
      return nullptr;
    }

    alloc_count_++;
    chan.reset(ch_ptr, free_func);
    return chan;
  }

  void Put(acldvppChannelDesc *desc, int32_t device_id) {
    std::lock_guard<std::mutex> lock(chan_list_lock_);
    auto &chan_list = device_chan_list_[device_id];
    chan_list.push_back(desc);
  }

  ~DvppChanMgr() {
    for (auto &chan_list_item : device_chan_list_) {
      for (auto ptr : chan_list_item.second) {
        acldvppDestroyChannel(ptr);
        acldvppDestroyChannelDesc(ptr);
      }
    }
  }

 private:
  std::mutex chan_list_lock_;
  std::unordered_map<int32_t, std::list<acldvppChannelDesc *>>
      device_chan_list_;
  std::atomic_uint64_t alloc_count_{0};
};

static std::shared_ptr<DvppChanMgr> g_dvpp_chan_mgr =
    std::make_shared<DvppChanMgr>();
std::shared_ptr<acldvppChannelDesc> GetDvppChannel(int32_t device_id) {
  return g_dvpp_chan_mgr->Get(device_id);
}

#endif  // ACL_ENABLE

};  // namespace imageprocess