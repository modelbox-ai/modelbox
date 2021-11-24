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


#ifndef MODELBOX_FLOWUNIT_NPPI_COLOR_CONVERTER_H_
#define MODELBOX_FLOWUNIT_NPPI_COLOR_CONVERTER_H_

#include <modelbox/base/status.h>
#include <nppi_color_conversion.h>
#include <functional>
#include <map>

class NppiColorConverter {
 public:
  NppiColorConverter();
  virtual ~NppiColorConverter();

  modelbox::Status CvtColor(const uint8_t *src, int32_t width, int32_t height,
                          uint8_t *dest, const std::string &pix_fmt);

 private:
  std::map<std::string,
           std::function<NppStatus(const uint8_t *const src[2],
                                   int32_t src_step, uint8_t *dest,
                                   int32_t dest_step, NppiSize size_roi)>>
      cvt_color_;
};

#endif  // MODELBOX_FLOWUNIT_NPPI_COLOR_CONVERTER_H_