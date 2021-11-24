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


#ifndef MODELBOX_FLOWUNIT_FACEALIGNMENTFLOWUNIT_CPU_H_
#define MODELBOX_FLOWUNIT_FACEALIGNMENTFLOWUNIT_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <algorithm>
#include <map>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME = "face_alignment";
constexpr const char *FLOWUNIT_DESC = "A cpu faceAlignment flowunit";
constexpr const char *DEVICE_TYPE = "cpu";

static const int KPS_NUM = 5;
static const int RGB_CHANNELS = 3;
static const uint32_t NET_INPUT_H_DEFAULT = 224;
static const uint32_t NET_INPUT_W_DEFAULT = 224;
static const float FACE_MEAN[3] = {123.68, 116.28, 103.03};
static const float FACE_VARIANCE[3] = {0.0171247, 0.0171247, 0.0171247};
static const std::vector<std::string> ALIGN_UNIT_IN_NAME = {"In_img", "In_kps"};
static const std::vector<std::string> ALIGN_UNIT_OUT_NAME = {"Aligned_img"};

typedef struct TagCoordinatePoint {
  float x = 0;
  float y = 0;
} CoordinatePoints;

typedef struct TagFaceKps {
  CoordinatePoints kps[KPS_NUM];
} FacePoints;

class FaceAlignmentFlowUnit : public modelbox::FlowUnit {
 public:
  FaceAlignmentFlowUnit();
  virtual ~FaceAlignmentFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close() { return modelbox::STATUS_OK; };

  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

 private:
  uint32_t net_input_h_{NET_INPUT_H_DEFAULT};
  uint32_t net_input_w_{NET_INPUT_W_DEFAULT};
  cv::Mat matrixA_;
  modelbox::Status CheckChannelSize(int port1Size, int port2Size);
  void InitRadiationMatrix();
  void EachFaceAlignment(const cv::Mat &image,
                         std::shared_ptr<FacePoints> &keypoints,
                         cv::Mat &aligned_image);
};

#endif  // MODELBOX_FLOWUNIT_FACEALIGNMENTFLOWUNIT_CPU_H_