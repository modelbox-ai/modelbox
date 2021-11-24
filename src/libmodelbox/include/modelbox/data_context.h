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

#ifndef MODELBOX_DATA_CONTEXT_H_
#define MODELBOX_DATA_CONTEXT_H_

#include <modelbox/buffer.h>
#include <modelbox/buffer_list.h>
#include <modelbox/index_buffer.h>
#include <modelbox/inner_event.h>
#include <modelbox/session_context.h>
#include <modelbox/statistics.h>
#include <modelbox/stream.h>

#include "modelbox/output_buffer.h"

#define CONFIG_NODES "nodes"
#define CONFIG_NODE "node."
#define CONFIG_FLOWUNIT "flowunit."

namespace modelbox {

class VirtualStream {
 public:
  VirtualStream(std::shared_ptr<BufferGroup> bg, int priority);
  virtual ~VirtualStream();
  bool IsClosed();
  /**
   * @brief Judge if the stream need close
   *
   */
  void Close();

  /**
   * @brief Get the Last Buffer Group object this should be shutdown
   *
   * @return std::shared_ptr<BufferGroup>
   */
  std::shared_ptr<BufferGroup> GetLastBufferGroup();

  /**
   * @brief Set the Session Context object .the SetSessionContext should be
   * deleted
   *
   * @param session_cxt
   */
  void SetSessionContext(std::shared_ptr<SessionContext> session_cxt);

  std::shared_ptr<SessionContext> GetSessionContext();

  /**
   * @brief Std::shared_ptr<IndexBufferList>
   * LabelIndexBuffer(std::shared_ptr<BufferList> index_buffer_list)
   *
   * @param index_buffer_list
   */
  void LabelIndexBuffer(std::shared_ptr<IndexBufferList> index_buffer_list);

  std::shared_ptr<BufferGroup> GetBufferGroup();

 private:
  uint32_t port_id_;
  std::shared_ptr<BufferGroup> stream_bg_;
  int priority_ = 0;
};

using FlowunitEventList =
    std::shared_ptr<std::vector<std::shared_ptr<FlowUnitInnerEvent>>>;

using InputData =
    std::unordered_map<std::string, std::shared_ptr<IndexBufferList>>;

class Node;

enum NotifyEvent { RECV_DATA, ERROR };

class ExternalData {
 public:
  virtual ~ExternalData() = default;
  virtual std::shared_ptr<BufferList> CreateBufferList() = 0;
  virtual Status Send(std::shared_ptr<BufferList> buffer_list) = 0;
  virtual std::shared_ptr<SessionContext> GetSessionContext() = 0;
  virtual Status SetOutputMeta(std::shared_ptr<DataMeta> meta) = 0;
  virtual Status Shutdown() = 0;
  virtual Status Close() = 0;

  virtual std::shared_ptr<Configuration> GetSessionConfig() = 0;
};

class InPort;
class ExternalDataImpl : public ExternalData {
 public:
  ExternalDataImpl(std::shared_ptr<InPort> port, std::shared_ptr<Device> device,
                   std::shared_ptr<StatisticsItem> graph_stats = nullptr);
  virtual ~ExternalDataImpl() = default;

  std::shared_ptr<BufferList> CreateBufferList() override;
  Status Send(std::shared_ptr<BufferList> buffer_list) override;

  std::shared_ptr<SessionContext> GetSessionContext() override;

  Status SetOutputMeta(std::shared_ptr<DataMeta> meta) override;

  Status Shutdown() override;

  Status Close() override;

  std::shared_ptr<Configuration> GetSessionConfig() override;

 private:
  std::shared_ptr<Device> device_;
  std::shared_ptr<InPort> ext_port_;
  std::shared_ptr<DataMeta> input_meta_;
  std::weak_ptr<SessionContext> session_context_;
  std::shared_ptr<VirtualStream> virtual_stream_;
};

class ExternalPort {
  virtual std::shared_ptr<ExternalData> Open() = 0;
};

enum class DataContextStatsType { NODE, SESSION, GRAPH };

class DataContext {
 public:
  virtual ~DataContext() = default;

  virtual std::shared_ptr<BufferList> Input(const std::string &port) const = 0;

  virtual std::shared_ptr<BufferList> Output(const std::string &port) = 0;

  virtual std::shared_ptr<BufferListMap> Input() const = 0;

  virtual std::shared_ptr<BufferListMap> Output() = 0;

  virtual std::shared_ptr<BufferList> External() = 0;

  virtual std::shared_ptr<FlowUnitEvent> Event() = 0;

  virtual bool HasError() = 0;

  virtual std::shared_ptr<FlowUnitError> GetError() = 0;

  virtual void SendEvent(std::shared_ptr<FlowUnitEvent> event) = 0;

  virtual void SetPrivate(const std::string &key,
                          std::shared_ptr<void> private_content) = 0;

  virtual std::shared_ptr<void> GetPrivate(const std::string &key) = 0;

  virtual const std::shared_ptr<DataMeta> GetInputMeta(
      const std::string &port) = 0;

  virtual const std::shared_ptr<DataMeta> GetInputGroupMeta(
      const std::string &port) = 0;

  virtual void SetOutputMeta(const std::string &port,
                             std::shared_ptr<DataMeta> data_meta) = 0;

  virtual std::shared_ptr<SessionContext> GetSessionContext() = 0;

  virtual std::shared_ptr<Configuration> GetSessionConfig() = 0;

  virtual std::shared_ptr<StatisticsItem> GetStatistics(
      DataContextStatsType type = DataContextStatsType::NODE) = 0;
};

class FlowUnitDataContext : public DataContext {
 public:
  FlowUnitDataContext(std::shared_ptr<BufferGroup> stream_bg, Node *node);

  virtual ~FlowUnitDataContext();

  virtual std::shared_ptr<BufferList> Input(const std::string &port) const override;

  virtual std::shared_ptr<BufferList> Output(const std::string &port) override;

  virtual std::shared_ptr<BufferListMap> Input() const override;

  virtual std::shared_ptr<BufferListMap> Output() override;

  virtual std::shared_ptr<BufferList> External() override;

  virtual std::shared_ptr<FlowUnitEvent> Event() override;

  virtual bool HasError() override;

  virtual std::shared_ptr<FlowUnitError> GetError() override;

  virtual void SetPrivate(const std::string &key,
                          std::shared_ptr<void> private_content) override;

  virtual std::shared_ptr<void> GetPrivate(const std::string &key) override;

  virtual void SendEvent(std::shared_ptr<FlowUnitEvent> event) override;

  virtual const std::shared_ptr<DataMeta> GetInputMeta(const std::string &port) override;

  virtual const std::shared_ptr<DataMeta> GetInputGroupMeta(
      const std::string &port)  override;

  virtual void SetOutputMeta(const std::string &port,
                             std::shared_ptr<DataMeta> data_meta) override;

  virtual std::shared_ptr<SessionContext> GetSessionContext() override;

  const std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
      &GetInputs() const;

  const std::unordered_map<std::string, std::shared_ptr<BufferList>>
      &GetOutputs() const;

  const std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
      &GetExternals() const;

  std::shared_ptr<StatisticsItem> GetStatistics(
      DataContextStatsType type) override;

  void SetOutput(
      const std::unordered_map<std::string, std::shared_ptr<BufferList>>
          &data_list);

  void SetStatus(Status status);

  Status GetStatus();

  Status GetLastStatus();

  bool IsOutputStreamError();

  virtual std::shared_ptr<Configuration> GetSessionConfig() override;

  void AddDestroyCallback(const std::function<void()> &func);

 protected:
  void UpdateInputInfo(std::shared_ptr<IndexBufferList> buffer_list);

  void UpdateInputDataMeta(std::string key,
                           std::shared_ptr<IndexBufferList> buffer_list);

  void UpdateErrorIndex(std::shared_ptr<InputData> input_data);

  void UpdateOutputDataMeta();
  virtual void SetInputData(std::shared_ptr<InputData> input_data) = 0;
  virtual std::shared_ptr<FlowUnitInnerEvent> GenerateSendEvent() = 0;
  virtual Status LabelData() = 0;
  virtual Status LabelError() = 0;
  virtual bool IsDataErrorVisible() = 0;
  virtual bool IsDataGroupPre() = 0;
  virtual bool IsDataGroupPost() = 0;
  virtual bool IsDataPre() = 0;
  virtual bool IsDataPost() = 0;
  virtual void ClearData() = 0;
  virtual void DealWithDataPreError(std::shared_ptr<FlowUnitError> error) = 0;
  virtual void DealWithProcessError(std::shared_ptr<FlowUnitError> error) = 0;
  virtual void DealWithDataError(std::shared_ptr<FlowUnitError> error) = 0;
  virtual Status GenerateOutData() = 0;
  virtual void CloseStreamIfNecessary() = 0;
  void UpdateDataPostFlag(bool need_to_datapost);
  virtual void UpdateStartFlag();

  bool IsSkippable();
  void SetSkippable(bool skippable);
  void FillEmptyOutput();

  Status process_status_;
  Status last_process_status_;

  std::weak_ptr<SessionContext> session_context_;

  std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>> input_;
  std::unordered_map<std::string, std::shared_ptr<BufferList>> output_;
  std::shared_ptr<OutputRings> output_data_;

  std::unordered_map<std::string, std::tuple<uint32_t, uint32_t>>
      input_error_index_;

  std::shared_ptr<FlowUnitError> output_stream_error_;

  bool start_flag_;
  bool end_flag_;
  bool closed_;
  bool is_finished_;
  bool is_skippable_;
  int32_t priority_;

  Node *node_;
  std::shared_ptr<BufferGroup> stream_index_;
  std::shared_ptr<FlowUnitInnerEvent> inner_event_;
  bool data_post_flag_;

  bool is_exception_visible_;

  bool is_input_meta_available_;

  bool is_input_group_meta_available_;

  bool is_output_meta_available_;

 private:
  friend class Node;
  friend class VirtualNode;
  friend class SingleNode;
  friend class FlowUnitGroup;
  friend class ExecutorDataContext;

  bool IsErrorStatus();

  void SetError(const std::shared_ptr<FlowUnitError> error);

  std::shared_ptr<FlowUnitError> GetInputError();

  void SetEvent(std::shared_ptr<FlowUnitEvent> event);

  Status AppendOutputMap(OutputIndexBuffer *output_map_index_buffer);

  bool IsFinished();

  void InitStatistic();

  std::vector<std::shared_ptr<Buffer>> external_;

  std::shared_ptr<FlowUnitError> error_;

  std::unordered_map<std::string, std::shared_ptr<DataMeta>> input_port_meta_;

  std::unordered_map<std::string, std::shared_ptr<DataMeta>> output_port_meta_;

  std::unordered_map<std::string, std::shared_ptr<void>> private_map_;

  std::shared_ptr<FlowUnitEvent> user_event_;

  std::shared_ptr<StatisticsItem> node_stats_;
  std::shared_ptr<StatisticsItem> session_stats_;
  std::shared_ptr<StatisticsItem> graph_stats_;

  std::list<std::function<void()>> destroy_callback_list_;
};

class NormalExpandFlowUnitDataContext : public FlowUnitDataContext {
 public:
  NormalExpandFlowUnitDataContext(std::shared_ptr<BufferGroup> stream_bg,
                                  Node *node);
  virtual ~NormalExpandFlowUnitDataContext() = default;

  void SetInputData(std::shared_ptr<InputData> input_data) override;

 private:
  bool IsDataGroupPre() override;
  bool IsDataGroupPost() override;
  bool IsDataPre() override;
  bool IsDataPost() override;
  bool IsDataErrorVisible() override;
  std::shared_ptr<FlowUnitInnerEvent> GenerateSendEvent() override;
  Status LabelData() override;
  Status LabelError() override;
  void ClearData() override;
  Status GenerateOutData() override;
  void DealWithProcessError(std::shared_ptr<FlowUnitError> error) override;
  void DealWithDataPreError(std::shared_ptr<FlowUnitError> error) override;
  void DealWithDataError(std::shared_ptr<FlowUnitError> error) override;
  void CloseStreamIfNecessary() override;
  void CloseExpandStream();
  void InsertPlaceholderToTheOutput();
  void SendEvent(std::shared_ptr<FlowUnitEvent> event) override;
  std::shared_ptr<VirtualStream> expand_level_stream_;
  std::set<uint32_t> backfill_set_;
};

class StreamExpandFlowUnitDataContext : public FlowUnitDataContext {
 public:
  StreamExpandFlowUnitDataContext(std::shared_ptr<BufferGroup> stream_bg,
                                  Node *node);
  virtual ~StreamExpandFlowUnitDataContext() = default;

  void ExpandNextStream();

  void SetInputData(std::shared_ptr<InputData> input_data) override;

  void UpdateCurrentOrder();

 private:
  bool IsDataGroupPre() override;
  bool IsDataGroupPost() override;
  bool IsDataPre() override;
  bool IsDataPost() override;
  bool IsDataErrorVisible() override;
  std::shared_ptr<FlowUnitInnerEvent> GenerateSendEvent() override;
  Status LabelData() override;
  Status LabelError() override;
  void ClearData() override;
  void DealWithProcessError(std::shared_ptr<FlowUnitError> error) override;
  void DealWithDataPreError(std::shared_ptr<FlowUnitError> error) override;
  void DealWithDataError(std::shared_ptr<FlowUnitError> error) override;
  void CloseStreamIfNecessary() override;

  void CloseExpandStream();
  void ExpandNewStream(std::shared_ptr<InputData> &input_data);
  void InsertPlaceholderToTheOutput();
  Status GenerateOutData() override;
  uint32_t current_expand_order_;
  uint32_t expand_cache_num_;

  std::shared_ptr<BufferGroup> current_expand_group_;
  std::shared_ptr<InputData> input_spilt_cache_;
  std::shared_ptr<VirtualStream> expand_level_stream_;
  std::set<uint32_t> backfill_set_;
};

class NormalCollapseFlowUnitDataContext : public FlowUnitDataContext {
 public:
  NormalCollapseFlowUnitDataContext(std::shared_ptr<BufferGroup> stream_bg,
                                    Node *node);
  virtual ~NormalCollapseFlowUnitDataContext() = default;
  void SetInputData(std::shared_ptr<InputData> input_data) override;

 private:
  bool IsDataGroupPre() override;
  bool IsDataGroupPost() override;
  bool IsDataPre() override;
  bool IsDataPost() override;
  bool IsDataErrorVisible() override;
  std::shared_ptr<FlowUnitInnerEvent> GenerateSendEvent() override;
  Status LabelData() override;
  Status LabelError() override;
  void ClearData() override;
  Status GenerateOutData() override;
  void DealWithProcessError(std::shared_ptr<FlowUnitError> error) override;
  void DealWithDataPreError(std::shared_ptr<FlowUnitError> error) override;
  void DealWithDataError(std::shared_ptr<FlowUnitError> error) override;
  void CloseStreamIfNecessary() override;
  void UpdateErrorIndex(std::shared_ptr<InputData> input_data);
  std::unordered_map<std::string, std::tuple<uint32_t, uint32_t>>
      input_group_error_index_;
  std::shared_ptr<FlowUnitError> output_group_stream_error_;
  std::set<uint32_t> backfill_set_;
};

class StreamCollapseFlowUnitDataContext : public FlowUnitDataContext {
 public:
  StreamCollapseFlowUnitDataContext(std::shared_ptr<BufferGroup> stream_bg,
                                    Node *node);
  virtual ~StreamCollapseFlowUnitDataContext() = default;

  void CollapseNextStream();
  void SetInputData(std::shared_ptr<InputData> input_data) override;
  void DealWithDataGroupPreError(std::shared_ptr<FlowUnitError> error);
  void UpdateCurrentOrder();
  void UpdateDataGroupPostFlag(bool flag);

 private:
  bool IsDataGroupPre() override;
  bool IsDataGroupPost() override;
  bool IsDataPre() override;
  bool IsDataPost() override;
  bool IsDataErrorVisible() override;
  std::shared_ptr<FlowUnitInnerEvent> GenerateSendEvent() override;
  Status LabelData() override;
  Status LabelError() override;
  void ClearData() override;

  Status GenerateOutData() override;
  const std::shared_ptr<DataMeta> GetInputGroupMeta(const std::string &port) override;
  void CollapseNewStream(std::shared_ptr<InputData> &input_data);
  void UpdateInputInfo(std::shared_ptr<IndexBufferList> buffer_list);
  void UpdateInputDataMeta(std::string key,
                           std::shared_ptr<IndexBufferList> buffer_list);
  void UpdateErrorIndex(std::shared_ptr<InputData> input_data);
  void CloseCollapseStream();

  void DealWithDataPreError(std::shared_ptr<FlowUnitError> error) override;

  void DealWithProcessError(std::shared_ptr<FlowUnitError> error) override;

  void DealWithDataError(std::shared_ptr<FlowUnitError> error) override;
  void CloseStreamIfNecessary() override;

  // this is used to decide the output stream is same as the input group stream
  bool is_group_stream_same_count_;

  bool group_start_flag_;

  bool group_end_flag_;

  // this is used when the flow process data_group_pre failed

  bool data_group_post_flag_;

  uint32_t current_collapse_order_;

  std::shared_ptr<FlowUnitError> output_group_stream_error_;

  std::shared_ptr<VirtualStream> collapse_level_stream_;

  std::unordered_map<std::string, std::shared_ptr<DataMeta>>
      input_group_port_meta_;

  std::unordered_map<std::string, std::tuple<uint32_t, uint32_t>>
      input_group_error_index_;

  std::unordered_map<uint32_t, std::shared_ptr<InputData>> input_cache_;
  std::set<uint32_t> backfill_set_;
};

class StreamFlowUnitDataContext : public FlowUnitDataContext {
 public:
  StreamFlowUnitDataContext(std::shared_ptr<BufferGroup> stream_bg, Node *node);
  virtual ~StreamFlowUnitDataContext() = default;
  void SetInputData(std::shared_ptr<InputData> input_data) override;

 private:
  bool IsDataGroupPre() override;
  bool IsDataGroupPost() override;
  bool IsDataPre() override;
  bool IsDataPost() override;
  bool IsDataErrorVisible() override;
  std::shared_ptr<FlowUnitInnerEvent> GenerateSendEvent() override;
  Status LabelData() override;
  Status LabelError() override;
  void ClearData() override;
  void CloseStream();
  void DealWithProcessError(std::shared_ptr<FlowUnitError> error) override;
  void DealWithDataPreError(std::shared_ptr<FlowUnitError> error) override;
  void DealWithDataError(std::shared_ptr<FlowUnitError> error) override;
  Status GenerateOutData() override;
  void CloseStreamIfNecessary() override;
  void UpdateStartFlag() override;

  bool is_stream_same_count_;
  std::shared_ptr<VirtualStream> same_level_stream_;
  std::vector<std::shared_ptr<BufferGroup>> buffer_index_;
  std::set<uint32_t> backfill_set_;
};

class NormalFlowUnitDataContext : public FlowUnitDataContext {
 public:
  NormalFlowUnitDataContext(std::shared_ptr<BufferGroup> stream_bg, Node *node);
  virtual ~NormalFlowUnitDataContext() = default;
  void SetInputData(std::shared_ptr<InputData> input_data) override;

 private:
  bool IsDataGroupPre() override;
  bool IsDataGroupPost() override;
  bool IsDataPre() override;
  bool IsDataPost() override;
  bool IsDataErrorVisible() override;
  std::shared_ptr<FlowUnitInnerEvent> GenerateSendEvent() override;
  Status LabelData() override;
  Status LabelError() override;
  void ClearData() override;
  void DealWithProcessError(std::shared_ptr<FlowUnitError> error) override;
  void DealWithDataPreError(std::shared_ptr<FlowUnitError> error) override;
  void DealWithDataError(std::shared_ptr<FlowUnitError> error) override;
  Status GenerateOutData() override;
  void CloseStreamIfNecessary() override;
  std::vector<std::shared_ptr<BufferGroup>> buffer_index_;
  std::set<uint32_t> backfill_set_;
};

class FlowUnitExecData;

class ExecutorDataContext : public DataContext {
 public:
  ExecutorDataContext(std::shared_ptr<FlowUnitDataContext> origin_ctx,
                      std::shared_ptr<FlowUnitExecData> data);
  virtual ~ExecutorDataContext() = default;

  virtual std::shared_ptr<BufferList> Input(
      const std::string &port) const override;

  virtual std::shared_ptr<BufferList> Output(const std::string &port) override;

  virtual std::shared_ptr<BufferListMap> Input() const override;

  virtual std::shared_ptr<BufferListMap> Output() override;

  virtual std::shared_ptr<BufferList> External() override;

  virtual bool HasError() override;

  virtual std::shared_ptr<FlowUnitError> GetError() override;

  virtual std::shared_ptr<FlowUnitEvent> Event() override;

  virtual void SendEvent(std::shared_ptr<FlowUnitEvent> event) override;

  virtual void SetPrivate(const std::string &key,
                          std::shared_ptr<void> private_content) override;

  virtual std::shared_ptr<void> GetPrivate(const std::string &key) override;

  virtual const std::shared_ptr<DataMeta> GetInputMeta(
      const std::string &port) override;

  virtual const std::shared_ptr<DataMeta> GetInputGroupMeta(
      const std::string &port) override;

  virtual void SetOutputMeta(const std::string &port,
                             std::shared_ptr<DataMeta> data_meta) override;

  virtual std::shared_ptr<SessionContext> GetSessionContext() override;

  void SetStatus(Status status);

  virtual std::shared_ptr<Configuration> GetSessionConfig() override;

  std::shared_ptr<StatisticsItem> GetStatistics(
      DataContextStatsType type) override;

 private:
  std::shared_ptr<FlowUnitDataContext> origin_ctx_;
  std::shared_ptr<FlowUnitExecData> data_;
};

}  // namespace modelbox
#endif
