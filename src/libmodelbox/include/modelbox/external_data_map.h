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

#ifndef MODELBOX_EXTERNAL_DATA_MAP_H_
#define MODELBOX_EXTERNAL_DATA_MAP_H_

#include <memory>

#include "modelbox/base/device.h"
#include "modelbox/error.h"
#include "modelbox/node.h"
#include "modelbox/port.h"
#include "modelbox/session.h"
#include "modelbox/statistics.h"

namespace modelbox {
class SessionContext;
class ExternalDataSelect;

class ExternalDataMap : public SessionIO {
 public:
  ExternalDataMap();
  ~ExternalDataMap() override;
  virtual std::shared_ptr<BufferList> CreateBufferList() = 0;
  Status SetOutputMeta(const std::string& port_name,
                       std::shared_ptr<DataMeta> meta) override = 0;
  Status Send(const std::string& port_name,
              std::shared_ptr<BufferList> buffer_list) override = 0;
  Status Recv(OutputBufferList& map_buffer_list,
              int32_t timeout = 0) override = 0;
  Status Close() override = 0;
  Status Shutdown() override = 0;
  virtual std::shared_ptr<SessionContext> GetSessionContext() = 0;
  virtual std::shared_ptr<Configuration> GetSessionConfig() = 0;
  virtual std::shared_ptr<FlowUnitError> GetLastError() = 0;

  virtual void SetPrivate(std::shared_ptr<void> ptr) = 0;

  virtual std::shared_ptr<void> GetPrivate() = 0;

  template <typename T>
  inline std::shared_ptr<T> GetPrivate() {
    return std::static_pointer_cast<T>(GetPrivate());
  }
};

class ExternalDataMapImpl
    : public ExternalDataMap,
      public std::enable_shared_from_this<ExternalDataMapImpl> {
 public:
  ExternalDataMapImpl(const std::shared_ptr<Node>& input_node,
                      const std::shared_ptr<Stream>& init_stream);

  ~ExternalDataMapImpl() override;

  std::shared_ptr<BufferList> CreateBufferList() override;

  Status SetOutputMeta(const std::string& port_name,
                       std::shared_ptr<DataMeta> meta) override;

  Status Send(const std::string& port_name,
              std::shared_ptr<BufferList> buffer_list) override;

  Status Recv(OutputBufferList& map_buffer_list, int32_t timeout = 0) override;

  Status Close() override;

  Status Shutdown() override;

  std::shared_ptr<SessionContext> GetSessionContext() override;

  std::shared_ptr<Configuration> GetSessionConfig() override;

  void SetPrivate(std::shared_ptr<void> ptr) override;

  std::shared_ptr<void> GetPrivate() override;

  void SetLastError(std::shared_ptr<FlowUnitError> error);

  std::shared_ptr<FlowUnitError> GetLastError() override;

  void SetSelector(const std::shared_ptr<ExternalDataSelect>& selector);

  bool GetReadyFlag();

  void PushGraphOutputBuffer(OutputBufferList& output);

 protected:
  void SessionEnd(std::shared_ptr<FlowUnitError> error = nullptr) override;

 private:
  Status PushToInputCache(const std::string& port_name,
                          const std::shared_ptr<BufferList>& buffer_list);

  void PopMachedInput(
      std::unordered_map<std::string, std::list<std::shared_ptr<Buffer>>>&
          matched_port_data,
      size_t& matched_data_size);

  Status SendMatchData(
      const std::unordered_map<std::string, std::list<std::shared_ptr<Buffer>>>&
          matched_port_data,
      size_t matched_data_size);

  // all extern output port stream inherit from init_stream
  std::shared_ptr<Stream> init_stream_;
  std::shared_ptr<BufferIndexInfo> root_buffer_;
  std::weak_ptr<Session> session_;
  std::weak_ptr<SessionContext> session_ctx_;

  std::shared_ptr<Node> graph_input_node_;
  std::shared_ptr<Device> graph_input_node_device_;
  std::unordered_map<std::string, std::shared_ptr<InPort>>
      graph_input_node_ports_;
  std::unordered_map<std::string, std::list<std::shared_ptr<Buffer>>>
      graph_input_ports_cache_;
  std::unordered_map<std::string, std::shared_ptr<Stream>>
      graph_input_ports_stream_;

  std::shared_ptr<FlowUnitError> last_error_;

  std::shared_ptr<BlockingQueue<OutputBufferList>> graph_output_cache_;
  std::weak_ptr<ExternalDataSelect> selector_;

  bool session_end_flag_{false};
  std::mutex session_state_lock_;

  bool close_flag_{false};
  bool shutdown_flag_{false};
  std::recursive_mutex close_state_lock_;
  std::shared_ptr<void> private_ptr_;
};

class ExternalDataSelect
    : public std::enable_shared_from_this<ExternalDataSelect> {
 public:
  ExternalDataSelect();
  virtual ~ExternalDataSelect();
  void RegisterExternalData(
      const std::shared_ptr<ExternalDataMap>& externl_data);
  void RemoveExternalData(const std::shared_ptr<ExternalDataMap>& externl_data);

  Status SelectExternalData(
      std::list<std::shared_ptr<ExternalDataMap>>& external_list,
      std::chrono::duration<long, std::milli> waittime =
          std::chrono::milliseconds(-1));

  bool IsExternalDataReady();

 private:
  friend class ExternalDataMapImpl;
  void NotifySelect();

  std::mutex external_list_lock_;
  std::list<std::shared_ptr<ExternalDataMapImpl>> external_list_;

  std::mutex data_ready_mtx_;
  std::condition_variable data_ready_cv_;
};
}  // namespace modelbox

#endif  // MODELBOX_EXTERNAL_DATA_MAP_H_