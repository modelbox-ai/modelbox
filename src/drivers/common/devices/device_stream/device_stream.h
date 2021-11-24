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


#ifndef MODELBOX_FLOWUNIT_DEVICE_STREAM_COMMON_H_
#define MODELBOX_FLOWUNIT_DEVICE_STREAM_COMMON_H_

#include "modelbox/base/device.h"
#include "modelbox/base/device_memory.h"
#include "modelbox/base/status.h"
#include "modelbox/flowunit.h"

namespace modelbox {

template <typename Memory>
void TravelDevMem(std::shared_ptr<modelbox::BufferListMap> input_buffer_list_map,
                  std::function<bool(std::shared_ptr<Memory>)> func) {
  for (auto &port : *input_buffer_list_map) {
    auto &buffer_list = port.second;
    auto dev_mem_list = buffer_list->GetAllBufferDeviceMemory();
    for (auto &dev_mem : dev_mem_list) {
      auto target_dev_mem = std::dynamic_pointer_cast<Memory>(dev_mem);
      if (target_dev_mem == nullptr) {
        continue;
      }

      auto ret = func(target_dev_mem);
      if (!ret) {
        return;
      }
    }
  }
}

template <typename Stream, typename Memory>
std::shared_ptr<Stream> GetDevSyncStream(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto input_buffer_list_map = data_ctx->Input();
  std::shared_ptr<Stream> first_stream;

  // Get first stream
  TravelDevMem<Memory>(input_buffer_list_map,
                       [&first_stream](std::shared_ptr<Memory> dev_mem) {
                         auto dev_stream = dev_mem->GetBindStream();
                         if (first_stream == nullptr && dev_stream != nullptr) {
                           first_stream = dev_stream;
                           return false;
                         }

                         return true;
                       });

  // Bind to same stream
  TravelDevMem<Memory>(
      input_buffer_list_map, [&first_stream](std::shared_ptr<Memory> dev_mem) {
        if (first_stream == nullptr) {
          // All dev mem has no stream, will create new stream
          auto status = dev_mem->BindStream();
          if (status != modelbox::STATUS_OK) {
            auto err_msg = "bind stream failed, " + status.WrapErrormsgs();
            MBLOG_ERROR << err_msg;
            return false;
          }

          first_stream = dev_mem->GetBindStream();
          return true;
        }

        if (first_stream != dev_mem->GetBindStream()) {
          // Sync different stream
          auto status = dev_mem->DetachStream();
          if (status != modelbox::STATUS_OK) {
            auto err_msg = "Detach stream failed, " + status.WrapErrormsgs();
            MBLOG_WARN << err_msg;
          }

          status = dev_mem->BindStream(first_stream);
          if (status != modelbox::STATUS_OK) {
            auto err_msg = "bind stream failed, " + status.WrapErrormsgs();
            MBLOG_WARN << err_msg;
          }
        }

        return true;
      });

  return first_stream;
};

template <typename Stream, typename Memory>
Status SetDevStream(std::shared_ptr<modelbox::DataContext> data_ctx,
                    const std::shared_ptr<Stream> &stream) {
  if (stream == nullptr) {
    return modelbox::STATUS_OK;
  }

  auto output_buffer_list_map = data_ctx->Output();
  TravelDevMem<Memory>(
      output_buffer_list_map, [stream](std::shared_ptr<Memory> dev_mem) {
        if (dev_mem->GetBindStream() == nullptr) {
          auto status = dev_mem->BindStream(stream);
          if (status != modelbox::STATUS_OK) {
            auto err_msg = "bind stream failed, " + status.WrapErrormsgs();
            MBLOG_WARN << err_msg;
          }
        }

        return true;
      });

  return modelbox::STATUS_OK;
};

template <typename Stream>
Status HoldMemory(std::shared_ptr<modelbox::DataContext> data_ctx,
                  const std::shared_ptr<Stream> &stream) {
  // Release input of this unit after stream stage completed
  // No need to bind output, it will pass to next unit
  // Refer to CudaStream/AscendStream->Bind() for detail
  auto input_buffer_list_map = data_ctx->Input();
  std::vector<std::shared_ptr<const DeviceMemory>> mems_to_hold;
  for (auto &item : *input_buffer_list_map) {
    auto &buffer_list = item.second;
    for (auto &buffer : *buffer_list) {
      mems_to_hold.push_back(buffer->GetDeviceMemory());
    }
  }

  auto ret = stream->Bind(mems_to_hold);
  if (!ret) {
    MBLOG_ERROR << "Bind mem for stream " << stream->Get() << " failed";
    return ret;
  }

  return modelbox::STATUS_OK;
}

};  // namespace modelbox

#endif  // MODELBOX_FLOWUNIT_DEVICE_STREAM_COMMON_H_