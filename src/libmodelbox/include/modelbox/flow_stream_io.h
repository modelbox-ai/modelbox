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
#ifndef FLOW_STREAM_IO_H_
#define FLOW_STREAM_IO_H_

#include <list>
#include <memory>
#include <string>

#include "modelbox/buffer.h"
#include "modelbox/external_data_map.h"

namespace modelbox {

class FlowStreamIO {
 public:
  FlowStreamIO(std::shared_ptr<ExternalDataMap> data_map);
  virtual ~FlowStreamIO();

  /**
   * @brief create a empty buffer on cpu device
   * @return cpu buffer
   **/
  std::shared_ptr<Buffer> CreateBuffer();

  /**
   * @brief Send buffer of this stream to flow
   * @param input_name input node name of flow
   * @param buffer buffer of this stream
   * @return Status
   **/
  Status Send(const std::string &input_name,
              const std::shared_ptr<Buffer> &buffer);
  /**
   * @brief Send buffer of this stream to flow
   * @param input_name input node name of flow
   * @param data data pointer
   * @param size data size
   * @return Status
   **/
  Status Send(const std::string &input_name, void *data, size_t size);

  /**
   * @brief recv buffer of this stream result from flow
   * @param output_name output node name of flow
   * @param buffer result buffer of this stream
   * @param timeout wait result timeout
   * @return Status
   **/
  Status Recv(const std::string &output_name, std::shared_ptr<Buffer> &buffer,
              long timeout = 0);

  /**
   * @brief recv buffer of this stream result from flow
   * @param output_name output node name of flow
   * @param timeout wait result timeout
   * @return buffer
   **/
  std::shared_ptr<Buffer> Recv(const std::string &output_name, long timeout);

  /**
   * @brief close input stream, mark stream end
   **/
  void CloseInput();

 private:
  std::shared_ptr<Device> device_;
  std::shared_ptr<ExternalDataMap> data_map_;
  std::map<std::string, std::list<std::shared_ptr<Buffer>>>
      port_data_cache_map_;
  Status status_;
};

}  // namespace modelbox
#endif  // FLOW_STREAM_IO_H_