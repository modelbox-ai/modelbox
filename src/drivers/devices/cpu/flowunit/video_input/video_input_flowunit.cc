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

#include "video_input_flowunit.h"

#include <securec.h>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

VideoInputFlowUnit::VideoInputFlowUnit(){};
VideoInputFlowUnit::~VideoInputFlowUnit(){};

modelbox::Status VideoInputFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  auto source_url = opts->GetString("source_url");
  auto repeat = opts->GetUint64("repeat", 1);
  // we need create new thread to send data to avoid stuck on queue
  auto write_data_func = [source_url, repeat, this]() {
    for (uint64_t i = 0; i < repeat; i++) {
      auto ext_data = this->CreateExternalData();
      if (!ext_data) {
        MBLOG_ERROR << "can not get external data.";
      }

      auto output_buf = ext_data->CreateBufferList();
      modelbox::TensorList output_tensor_list(output_buf);
      output_tensor_list.BuildFromHost<unsigned char>(
          {1, {source_url.size() + 1}}, (void *)source_url.data(),
          source_url.size() + 1);

      auto data_meta = std::make_shared<modelbox::DataMeta>();
      data_meta->SetMeta("source_url",
                         std::make_shared<std::string>(source_url));

      ext_data->SetOutputMeta(data_meta);

      auto status = ext_data->Send(output_buf);
      if (!status) {
        MBLOG_ERROR << "external data send buffer list failed:" << status;
      }

      status = ext_data->Close();
      if (!status) {
        MBLOG_ERROR << "external data close failed:" << status;
      }
    }
  };

  std::thread write_data_thread(write_data_func);
  write_data_thread.detach();
  return modelbox::STATUS_OK;
}
modelbox::Status VideoInputFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status VideoInputFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto output_buf = data_ctx->Output("out_video_url");
  std::vector<size_t> shape(1, 1);
  output_buf->Build(shape);
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(VideoInputFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Video");
  desc.AddFlowUnitOutput({"out_video_url"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("source_url", "string", true,
                                                  "", "the video  source url"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
