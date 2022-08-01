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

#ifndef MODELBOX_FLOWUNIT_IMAGE_PROCESS_COMMON_H_
#define MODELBOX_FLOWUNIT_IMAGE_PROCESS_COMMON_H_

#include <modelbox/base/status.h>
#include <modelbox/flowunit.h>

#include <string>

#ifdef ACL_ENABLE

#ifndef ENABLE_DVPP_INTERFACE
#define ENABLE_DVPP_INTERFACE
#endif

#include <acl/ops/acl_dvpp.h>

#endif  // ACL_ENABLE

namespace imageprocess {

typedef struct RoiBox {
  int32_t x, y, w, h;
} RoiBox;

class ImageShape {
 public:
  ImageShape(int32_t img_width, int32_t img_height, int32_t img_width_stride,
             int32_t img_height_stride)
      : width{img_width},
        height{img_height},
        width_stride{img_width_stride},
        height_stride{img_height_stride} {}

  virtual ~ImageShape() = default;

  int32_t width{0};
  int32_t height{0};
  int32_t width_stride{0};
  int32_t height_stride{0};
};

int32_t align_up(int32_t num, int32_t align);

modelbox::Status GetImageBytes(const std::string &pix_fmt, size_t pix_num,
                               size_t &img_bytes);

modelbox::Status GetImageBytes(const std::string &pix_fmt, int32_t width,
                               int32_t height, size_t &img_bytes);

modelbox::Status GetWidthStride(const std::string &pix_fmt, int32_t width,
                                int32_t &width_stride);

modelbox::Status GetImageBytesByStride(const std::string &pix_fmt,
                                       int32_t width_stride,
                                       int32_t height_stride,
                                       size_t &img_bytes);

modelbox::Status GetImgParam(const std::shared_ptr<modelbox::Buffer> &img,
                             std::string &pix_fmt, int32_t &img_width,
                             int32_t &img_height, int32_t &img_width_stride,
                             int32_t &img_height_stride);

modelbox::Status CheckImageStride(const std::string &pix_fmt,
                                  int32_t img_width_stride,
                                  int32_t expect_w_align,
                                  int32_t img_height_stride,
                                  int32_t expect_h_align, size_t img_size);

bool CheckRoiBoxVaild(const RoiBox *bbox, int32_t image_width,
                                    int32_t image_height);

#ifdef ACL_ENABLE

const int32_t ASCEND_WIDTH_ALIGN = 16;
const int32_t ASCEND_HEIGHT_ALIGN = 2;

modelbox::Status InitDvppChannel(
    std::shared_ptr<acldvppChannelDesc> &chan_desc);

enum class ImgDescDestroyFlag { DESC_AND_BUFFER, DESC_ONLY, NONE };

std::shared_ptr<acldvppPicDesc> CreateImgDesc(
    size_t img_size, const std::string &pix_fmt, const ImageShape &shape,
    ImgDescDestroyFlag flag = ImgDescDestroyFlag::DESC_AND_BUFFER);

std::shared_ptr<acldvppPicDesc> CreateImgDesc(
    size_t img_size, void *img_buffer, const std::string &pix_fmt,
    const ImageShape &shape,
    ImgDescDestroyFlag flag = ImgDescDestroyFlag::DESC_AND_BUFFER);

modelbox::Status FillImgDescData(
    std::shared_ptr<acldvppPicDesc> &img_desc,
    std::shared_ptr<modelbox::Buffer> &image_buffer, aclrtStream stream);

modelbox::Status SetOutImgMeta(std::shared_ptr<modelbox::Buffer> &out_image,
                               const std::string &out_pix_fmt,
                               std::shared_ptr<acldvppPicDesc> &out_img_desc);

std::shared_ptr<acldvppChannelDesc> GetDvppChannel(int32_t device_id);

#endif  // ACL_ENABLE

};  // namespace imageprocess

#endif