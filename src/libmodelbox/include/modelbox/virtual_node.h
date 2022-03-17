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

#ifndef MODELBOX_VIRTUAL_NODE_H_
#define MODELBOX_VIRTUAL_NODE_H_

#include <chrono>

#include "modelbox/base/device.h"
#include "modelbox/node.h"
#include "modelbox/statistics.h"

namespace modelbox {

class ExternalDataMap {
 public:
  virtual ~ExternalDataMap() = default;
  virtual std::shared_ptr<BufferList> CreateBufferList() = 0;
  virtual Status Send(std::string port_name,
                      std::shared_ptr<BufferList> buffer_list) = 0;
  virtual Status Recv(OutputBufferList& map_buffer_list, int timeout = 0) = 0;
  virtual Status Close() = 0;
  virtual std::shared_ptr<SessionContext> GetSessionContext() = 0;
  virtual Status Shutdown() = 0;
  virtual Status SetOutputMeta(std::string port_name,
                               std::shared_ptr<DataMeta> meta) = 0;
  virtual std::shared_ptr<FlowUnitError> GetLastError() = 0;

  virtual std::shared_ptr<Configuration> GetSessionConfig() = 0;
};

class OutputUnmatchVirtualNode;
class InputVirtualNode;
class ExternalDataSelect;
class VirtualStream;
class ExternalDataMapImpl
    : public ExternalDataMap,
      public std::enable_shared_from_this<ExternalDataMapImpl> {
 public:
  ExternalDataMapImpl(std::shared_ptr<NodeBase> input_node,
                      std::shared_ptr<NodeBase> output_node,
                      std::shared_ptr<StatisticsItem> graph_stats = nullptr);
  virtual ~ExternalDataMapImpl() = default;
  std::shared_ptr<BufferList> CreateBufferList() override;

  Status Send(std::string port_name,
              std::shared_ptr<BufferList> buffer_list) override;

  // if we have no data to recv return STATUS_SUCESS else return STATUS_CONTINUE
  Status Recv(OutputBufferList& map_buffer_list, int timeout = 0) override;

  // unusual stop the stream
  Status Close() override;

  std::shared_ptr<SessionContext> GetSessionContext() override;

  Status SetOutputMeta(std::string port_name,
                       std::shared_ptr<DataMeta> meta) override;

  // normal close the stream
  Status Shutdown() override;

  std::shared_ptr<FlowUnitError> GetLastError() override;

  std::shared_ptr<Configuration> GetSessionConfig() override;

 private:
  friend class SessionContext;
  friend class Task;
  friend class Graph;
  friend class ExternalDataSelect;

  void Init();
  Status SendData(OriginDataMap& data);
  void SetEndError(std::shared_ptr<FlowUnitError> error);
  Status SetOutputBuffer(OutputBufferList& output);
  void UpdateInputMeta(std::string port_name,
                       std::shared_ptr<IndexBufferList> index_buffer_list);
  void SetSelector(std::shared_ptr<ExternalDataSelect> selector);
  bool GetReadyFlag();
  void UnbindSession();

  // lock_ protcet virtual_stream_
  std::mutex lock_;
  std::shared_ptr<Device> device_;
  std::shared_ptr<FlowUnitError> error_;
  bool end_flag_;
  std::unordered_map<std::string, std::shared_ptr<InPort>> input_ports_;
  std::shared_ptr<BlockingQueue<OutputBufferList>> output_buffer_cache_;
  std::weak_ptr<SessionContext> session_context_;
  std::shared_ptr<VirtualStream> virtual_stream_;
  std::unordered_map<std::string, std::list<std::shared_ptr<Buffer>>>
      input_buffer_cache_;
  std::unordered_map<std::string, std::shared_ptr<DataMeta>> input_meta_;
  std::unordered_map<std::string, std::shared_ptr<DataMeta>> output_meta_;

  std::weak_ptr<ExternalDataSelect> selector_;
};

class ExternalDataSelect
    : public std::enable_shared_from_this<ExternalDataSelect> {
 public:
  ExternalDataSelect();
  virtual ~ExternalDataSelect();
  void RegisterExternalData(std::shared_ptr<ExternalDataMap> externl_data);
  void RemoveExternalData(std::shared_ptr<ExternalDataMap>& externl_data);

  Status SelectExternalData(
      std::list<std::shared_ptr<ExternalDataMap>>& external_list,
      std::chrono::duration<long, std::milli> timeout =
          std::chrono::milliseconds(-1));

 private:
  friend class ExternalDataMapImpl;
  void NotifySelect();
  bool IsExtenalDataReady();
  std::list<std::shared_ptr<ExternalDataMapImpl>> external_list_;
  std::mutex mtx_;
  std::condition_variable cv_;
};

class InputVirtualNode : public DataMatcherNode {
 public:
  InputVirtualNode(const std::string& unit_device_name,
                   const std::string& unit_device_id,
                   std::shared_ptr<DeviceManager> device_manager);

  virtual ~InputVirtualNode() override;

  Status Init(const std::set<std::string>& input_ports,
              const std::set<std::string>& output_ports,
              std::shared_ptr<Configuration> config) override;

  /**
   * @brief Open the Node object
   *
   */
  Status Open() override;

  /**
   * @brief The node main function
   *
   * @param type run type
   * @return Status
   */
  Status Run(RunType type) override;

  bool ExternalToOutput(InputIndexBuffer& ext_buffer,
                        OutputIndexBuffer& output);

  std::shared_ptr<Device> GetDevice() override;

 private:
  Status RecvExternalData(InputIndexBuffer& input_buffer);
  InputIndexBuffer CreateExternalBuffer();
  std::shared_ptr<DeviceManager> device_mgr_;
  std::string device_name_;
  std::string device_id_;
};

class OutputVirtualNode : public DataMatcherNode {
 public:
  OutputVirtualNode(const std::string& device_name,
                    const std::string& device_id,
                    std::shared_ptr<DeviceManager> device_manager);

  virtual ~OutputVirtualNode();
  Status Init(const std::set<std::string>& input_port_names,
              const std::set<std::string>& output_port_names,
              std::shared_ptr<Configuration> config) override;

  /**
   * @brief Open the Node object
   *
   */
  Status Open() override;

  /**
   * @brief The node main function
   *
   * @param type run type
   * @return Status
   */
  Status Run(RunType type) override;

  std::shared_ptr<Device> GetDevice() override;

 private:
  Status RecvData(InputIndexBuffer& input_buffer);
  std::shared_ptr<DeviceManager> device_mgr_;
  std::string device_name_;
  std::string device_id_;
};

class MultiLevelCache {
 public:
  MultiLevelCache();

  virtual ~MultiLevelCache();

  void PushBack(std::shared_ptr<IndexBuffer>& buffer_vector);

  /**
   * @brief Pop out the cache buffer vector
   *
   * @param buffer_vector
   * @return true if the cache is not empty
   * @return false if the cache is empty
   */
  bool PopOut(std::vector<std::shared_ptr<Buffer>>& buffer_vector);

  bool PopOneGroup(std::string cur_key,
                   std::vector<std::shared_ptr<Buffer>>& buffer_vector);
  bool UpdateNextGroup();

  std::shared_ptr<FlowUnitError> GetError();

 private:
  uint32_t order_;
  std::map<std::shared_ptr<BufferGroup>,
           std::vector<std::shared_ptr<IndexBuffer>>>
      cache_;
  std::vector<uint32_t> cur_order_seq_;
  std::unordered_map<std::string, std::shared_ptr<BufferGroup>> key_bg_map_;
  std::shared_ptr<BufferGroup> cur_buffer_group_;
  std::shared_ptr<FlowUnitError> error_;
};

class OutputUnmatchVirtualNode : public NodeBase {
 public:
  OutputUnmatchVirtualNode(const std::string& device_name,
                           const std::string& device_id,
                           std::shared_ptr<DeviceManager> device_manager);

  virtual ~OutputUnmatchVirtualNode();

  Status Init(const std::set<std::string>& input_port_names,
              const std::set<std::string>& output_port_names,
              std::shared_ptr<Configuration> config) override;

  /**
   * @brief Open the Node object
   *
   */
  Status Open() override;

  /**
   * @brief The node main function
   *
   * @param type run type
   * @return Status
   */
  Status Run(RunType type) override;

  std::shared_ptr<Device> GetDevice() override;

 private:
  std::shared_ptr<DeviceManager> device_mgr_;
  std::string device_name_;
  std::string device_id_;
  std::unordered_map<
      std::shared_ptr<SessionContext>,
      std::unordered_map<std::string, std::shared_ptr<MultiLevelCache>>>
      cache_map_;
  std::unordered_map<std::shared_ptr<SessionContext>, uint32_t>
      session_sucess_count_;
};

}  // namespace modelbox
#endif  // MODELBOX_VIRTUAL_NODE_H_
