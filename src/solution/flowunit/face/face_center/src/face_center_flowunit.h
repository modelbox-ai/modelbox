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


#ifndef MODELBOX_FLOWUNIT_FACE_CENTER_FLOWUNIT_CPU_H_
#define MODELBOX_FLOWUNIT_FACE_CENTER_FLOWUNIT_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

typedef struct tagCoordinatePoint {
  float x = 0;
  float y = 0;
} CoordinatePoints;

typedef struct tagCenterFace {
  CoordinatePoints pt1;
  CoordinatePoints pt2;
  float score;
  int cls{1};  // default = 1;
  CoordinatePoints kps[5];
} CenterFaces;

typedef struct tagCenterKps {
  CoordinatePoints kps[5];
} CenterKeyPoints;

typedef struct tagImageShape {
  int w;
  int h;
  int c;
  int n;
} ImageShape;

constexpr const char *FLOWUNIT_NAME = "face_center";
constexpr const char *FLOWUNIT_DESC = "A cpu face_center flowunit";
static const int OUTPUT_CHANNEL = 4;
static const int INPUT_SHAPE_SIZE = 3;
static const float SCORE_THRESHOLD = 0.4;
static const float NMS_THRESHOLD = 0.6;
static const int ORIGINAL_IMAGE_W_DEFAULT = 2560;
static const int ORIGINAL_IMAGE_H_DEFAULT = 1440;
static const int NET_INPUT_W_DEFAULT = 640;
static const int NET_INPUT_H_DEFAULT = 352;
static const std::vector<std::string> CENTER_UNIT_IN_NAME = {
    "sigmoid_blob1", "conv_blob60", "conv_blob62", "conv_blob64"};
static const std::vector<std::string> CENTER_UNIT_OUT_NAME = {"Out_1", "Out_2"};

class FaceCenterFlowUnit : public modelbox::FlowUnit {
 public:
  FaceCenterFlowUnit();
  virtual ~FaceCenterFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close() { return modelbox::STATUS_OK; };

  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

 private:
  modelbox::Status InitCenterFlowunit(void);
  bool CheckShape(ImageShape shape, const std::vector<size_t> &input);
  void computeCenterBBox(
      std::vector<std::shared_ptr<modelbox::BufferList>> &infer_layers,
      std::vector<CenterFaces> &bboxes, int batch_idx);

  modelbox::Status GetInferLayers(
      std::shared_ptr<modelbox::DataContext> data_ctx,
      std::vector<std::shared_ptr<modelbox::BufferList>> &infer_layers);

  std::vector<CenterFaces> ComputerBox(
      std::vector<std::shared_ptr<modelbox::BufferList>> &infer_layers,
      int batch_idx);

  modelbox::Status SendBox(
      std::shared_ptr<modelbox::DataContext> data_ctx,
      const std::vector<std::vector<CenterFaces>> &out_bboxes);

  modelbox::Status SendKeyPoints(
      std::shared_ptr<modelbox::DataContext> data_ctx,
      const std::vector<std::vector<CenterFaces>> &out_bboxes);

  int original_img_w_{ORIGINAL_IMAGE_W_DEFAULT};
  int original_img_h_{ORIGINAL_IMAGE_H_DEFAULT};
  int net_input_h_{NET_INPUT_H_DEFAULT};
  int net_input_w_{NET_INPUT_W_DEFAULT};

  int net_output_h_;
  int net_output_w_;

  int feature_size_;
  int w_padding_;
  int h_padding_;

  float img_scale_;
  std::vector<ImageShape> out_shape_;
};

#endif  // MODELBOX_FLOWUNIT_FACE_CENTER_FLOWUNIT_CPU_H_