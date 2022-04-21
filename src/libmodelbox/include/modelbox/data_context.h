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
#include <modelbox/error.h>
#include <modelbox/inner_event.h>
#include <modelbox/match_stream.h>
#include <modelbox/session.h>
#include <modelbox/session_context.h>
#include <modelbox/statistics.h>
#include <modelbox/stream.h>

#include <unordered_set>

#define CONFIG_NODES "nodes"
#define CONFIG_NODE "node."
#define CONFIG_FLOWUNIT "flowunit."

namespace modelbox {

using FlowunitEventList =
    std::shared_ptr<std::vector<std::shared_ptr<FlowUnitInnerEvent>>>;

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
                   std::shared_ptr<Stream> init_stream);
  virtual ~ExternalDataImpl();

  std::shared_ptr<BufferList> CreateBufferList() override;
  Status Send(std::shared_ptr<BufferList> buffer_list) override;

  std::shared_ptr<SessionContext> GetSessionContext() override;

  Status SetOutputMeta(std::shared_ptr<DataMeta> meta) override;

  Status Shutdown() override;

  Status Close() override;

  std::shared_ptr<Configuration> GetSessionConfig() override;

 private:
  void SendCacheBuffer();
  bool is_closed_{false};

  std::shared_ptr<BufferIndexInfo> root_buffer_;

  std::shared_ptr<InPort> ext_port_;
  std::shared_ptr<Device> device_;
  std::shared_ptr<Stream> input_stream_;
  std::weak_ptr<Session> session_;
  std::weak_ptr<SessionContext> session_ctx_;

  std::shared_ptr<DataMeta> output_meta_;
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

class FlowUnitDataContext : public DataContext, public SessionStateListener {
  // Implement interface DataContext
 public:
  FlowUnitDataContext(Node *node, MatchKey *data_ctx_match_key,
                      std::shared_ptr<Session> session);

  virtual ~FlowUnitDataContext();

  std::shared_ptr<BufferList> Input(const std::string &port) const override;

  std::shared_ptr<BufferList> Output(const std::string &port) override;

  std::shared_ptr<BufferListMap> Input() const override;

  std::shared_ptr<BufferListMap> Output() override;

  std::shared_ptr<BufferList> External() override;

  void SetEvent(std::shared_ptr<FlowUnitEvent> event);

  std::shared_ptr<FlowUnitEvent> Event() override;

  bool HasError() override;

  std::shared_ptr<FlowUnitError> GetError() override;

  void SetPrivate(const std::string &key,
                  std::shared_ptr<void> private_content) override;

  std::shared_ptr<void> GetPrivate(const std::string &key) override;

  void SendEvent(std::shared_ptr<FlowUnitEvent> event) override;

  const std::shared_ptr<DataMeta> GetInputMeta(
      const std::string &port) override;

  const std::shared_ptr<DataMeta> GetInputGroupMeta(
      const std::string &port) override;

  void SetOutputMeta(const std::string &port,
                     std::shared_ptr<DataMeta> data_meta) override;

  std::shared_ptr<SessionContext> GetSessionContext() override;

  std::shared_ptr<Configuration> GetSessionConfig() override;

  std::shared_ptr<StatisticsItem> GetStatistics(
      DataContextStatsType type) override;

  // common function for FlowUnitDataContext
 public:
  const std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
      &GetInputs() const;

  const std::unordered_map<std::string, std::shared_ptr<BufferList>>
      &GetOutputs() const;

  const std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
      &GetExternals() const;

  void SetOutput(
      const std::unordered_map<std::string, std::shared_ptr<BufferList>>
          &data_list);

  void SetStatus(Status status);

  Status GetStatus();

  Status GetLastStatus();

  bool IsErrorStatus();

  void AddDestroyCallback(const std::function<void()> &func);

  bool IsDataErrorVisible();

  bool IsFinished();

  Status PopOutputData(PortDataMap &stream_data_map);

  std::unordered_map<std::string, std::shared_ptr<DataMeta>>
  GetOutputPortStreamMeta();

  bool IsSkippable();

  void SetSkippable(bool skippable);

  /**
   * @brief call after flowunit process
   **/
  Status PostProcess();

  std::shared_ptr<Session> GetSession();

  void NotifySessionClose() override;

  // would be different in specify FlowUnitDataContext
 public:
  // buffers in stream_data_map is in order
  virtual void WriteInputData(std::shared_ptr<PortDataMap> stream_data_map);

  virtual std::shared_ptr<FlowUnitInnerEvent> GenerateSendEvent() {
    return nullptr;
  }

  virtual bool IsDataPre() { return false; }
  virtual bool IsDataPost() { return false; }

  virtual void DealWithDataError();
  virtual void DealWithDataPreError(std::shared_ptr<FlowUnitError> error);
  virtual void DealWithProcessError(std::shared_ptr<FlowUnitError> error);

  /**
   * @brief call after flowunit group run
   **/
  virtual void UpdateProcessState();
  virtual void ClearData();

 protected:
  virtual void UpdateBufferIndexInfo(
      std::shared_ptr<BufferIndexInfo> cur_buffer,
      std::shared_ptr<BufferIndexInfo> parent_buffer){};

  virtual bool SkipInheritInputToMatchNode() { return false; }

  void SetCurrentInputData(std::shared_ptr<PortDataMap> stream_data_map);

  virtual void UpdateInputInfo();

  virtual Status GenerateOutputPlaceholder();

  virtual Status GenerateOutput();

  virtual Status AppendEndFlag();

  virtual bool NeedStreamEndFlag() { return false; };

  void FillPlaceholderOutput(bool from_valid_input = false,
                             bool same_with_input_num = true);

  void FillErrorOutput(std::shared_ptr<FlowUnitError> error,
                       bool same_with_input_num = true);

  bool HasValidOutput();

  size_t GetOutputBufferNum();

  virtual Status CheckOutputData() { return STATUS_OK; };

  bool IsContinueProcess();

  Status process_status_{STATUS_OK};
  Status last_process_status_{STATUS_OK};

  MatchKey *data_ctx_match_key_{nullptr};
  std::shared_ptr<Session> session_;
  std::weak_ptr<SessionContext> session_context_;

  // total input
  std::shared_ptr<PortDataMap> cur_input_;
  // valid data for flowunit process
  PortDataMap cur_input_valid_data_;
  // empty for drop, empty for condition
  PortDataMap cur_input_placeholder_;
  // end for one stream, empty buffer
  PortDataMap cur_input_end_flag_;
  // flowunit output
  std::unordered_map<std::string, std::shared_ptr<BufferList>>
      cur_output_valid_data_;
  // empty for drop, empty for condition
  PortDataMap cur_output_placeholder_;
  // total output
  PortDataMap cur_output_;

  Node *node_;

  // state for ctx
  bool is_exception_visible_{false};
  bool is_finished_{false};  // will not process this data ctx again

  // state for stream
  bool is_empty_stream{false};  // end_flag is first buffer of stream
  bool end_flag_received_{false};
  size_t input_stream_max_buffer_count_{0};
  size_t input_stream_cur_buffer_count_{0};

  // state for single run
  bool is_skippable_{false};  // no data
  std::mutex wait_user_events_lock_;
  std::unordered_set<std::shared_ptr<FlowUnitEvent>>
      wait_user_events_;  // user send event, wait to process

  bool input_has_stream_start_{false};
  bool input_has_stream_end_{false};

  std::shared_ptr<FlowUnitError> error_;

 private:
  void InitStatistic();

  Status UpdateOutputIndexInfo();

  std::shared_ptr<BufferProcessInfo> GetCurNodeProcessInfo(
      std::shared_ptr<BufferIndexInfo> index_info);

  std::unordered_map<std::string, std::shared_ptr<DataMeta>> input_port_meta_;
  std::unordered_map<std::string, std::shared_ptr<DataMeta>> output_port_meta_;

  std::unordered_map<std::string, std::shared_ptr<void>> private_map_;

  std::shared_ptr<FlowUnitEvent> user_event_;
  PortDataMap cur_event_input_data_;  // record for event

  std::shared_ptr<StatisticsItem> node_stats_;
  std::shared_ptr<StatisticsItem> session_stats_;
  std::shared_ptr<StatisticsItem> graph_stats_;

  std::list<std::function<void()>> destroy_callback_list_;
};

class NormalFlowUnitDataContext : public FlowUnitDataContext {
 public:
  NormalFlowUnitDataContext(Node *node, MatchKey *data_ctx_match_key,
                            std::shared_ptr<Session> session);
  virtual ~NormalFlowUnitDataContext() = default;

  void SendEvent(std::shared_ptr<FlowUnitEvent> event) override {
    // not support user send event
  }

  void UpdateProcessState() override;

 protected:
  bool NeedStreamEndFlag() override;

  void UpdateBufferIndexInfo(
      std::shared_ptr<BufferIndexInfo> cur_buffer,
      std::shared_ptr<BufferIndexInfo> parent_buffer) override;
};

class LoopNormalFlowUnitDataContext : public NormalFlowUnitDataContext {
 public:
  LoopNormalFlowUnitDataContext(Node *node, MatchKey *data_ctx_match_key,
                                std::shared_ptr<Session> session);
  virtual ~LoopNormalFlowUnitDataContext() = default;

 protected:
  Status GenerateOutput() override;

  Status AppendEndFlag() override;

  Status CheckOutputData() override;

  std::string output_port_for_this_loop_;

  PortDataMap cached_output_placeholder_;  // send cache after this loop decide

  PortDataMap cached_input_end_flag_;  // process after this loop decide
};

class StreamFlowUnitDataContext : public FlowUnitDataContext {
 public:
  StreamFlowUnitDataContext(Node *node, MatchKey *data_ctx_match_key,
                            std::shared_ptr<Session> session);
  virtual ~StreamFlowUnitDataContext() = default;

  bool IsDataPre() override;
  bool IsDataPost() override;

  void DealWithDataPreError(std::shared_ptr<FlowUnitError> error) override;

  void UpdateProcessState() override;

 protected:
  bool NeedStreamEndFlag() override;

  void UpdateBufferIndexInfo(
      std::shared_ptr<BufferIndexInfo> cur_buffer,
      std::shared_ptr<BufferIndexInfo> parent_buffer) override;

  PortDataMap cached_input_end_flag_;  // process after output stream end
};

class NormalExpandFlowUnitDataContext : public FlowUnitDataContext {
 public:
  NormalExpandFlowUnitDataContext(Node *node, MatchKey *data_ctx_match_key,
                                  std::shared_ptr<Session> session);

  virtual ~NormalExpandFlowUnitDataContext() = default;

  void DealWithDataPreError(std::shared_ptr<FlowUnitError> error) override;

  void UpdateProcessState() override;

 protected:
  bool NeedStreamEndFlag() override;

  void UpdateBufferIndexInfo(
      std::shared_ptr<BufferIndexInfo> cur_buffer,
      std::shared_ptr<BufferIndexInfo> parent_buffer) override;
};

class StreamExpandFlowUnitDataContext : public FlowUnitDataContext {
 public:
  StreamExpandFlowUnitDataContext(Node *node, MatchKey *data_ctx_match_key,
                                  std::shared_ptr<Session> session);
  virtual ~StreamExpandFlowUnitDataContext() = default;

  void WriteInputData(std::shared_ptr<PortDataMap> stream_data_map) override;

  void ExpandNextBuffer();

  bool IsDataPre() override;
  bool IsDataPost() override;

  std::shared_ptr<FlowUnitInnerEvent> GenerateSendEvent() override;

  void DealWithDataPreError(std::shared_ptr<FlowUnitError> error) override;

  void UpdateProcessState() override;

 protected:
  bool NeedStreamEndFlag() override;

  void UpdateBufferIndexInfo(
      std::shared_ptr<BufferIndexInfo> cur_buffer,
      std::shared_ptr<BufferIndexInfo> parent_buffer) override;

 private:
  // only read one buffer each process
  std::list<std::shared_ptr<PortDataMap>> stream_data_cache_;
  size_t cur_data_pose_in_first_cache_{0};
  size_t cur_expand_buffer_index_{0};
  bool cur_expand_buffer_index_received_{false};
};

class NormalCollapseFlowUnitDataContext : public FlowUnitDataContext {
 public:
  NormalCollapseFlowUnitDataContext(Node *node, MatchKey *data_ctx_match_key,
                                    std::shared_ptr<Session> session);
  virtual ~NormalCollapseFlowUnitDataContext() = default;

  void SendEvent(std::shared_ptr<FlowUnitEvent> event) override {
    // not support user send event
  }

  bool IsDataPre() override;
  bool IsDataPost() override;

  void DealWithDataPreError(std::shared_ptr<FlowUnitError> error) override;
  void DealWithProcessError(std::shared_ptr<FlowUnitError> error) override;

  void UpdateProcessState() override;

 protected:
  bool SkipInheritInputToMatchNode() override { return true; };

  Status GenerateOutputPlaceholder() override;

  bool NeedStreamEndFlag() override;

  Status CheckOutputData() override;

  void UpdateBufferIndexInfo(
      std::shared_ptr<BufferIndexInfo> cur_buffer,
      std::shared_ptr<BufferIndexInfo> parent_buffer) override;

 private:
  size_t output_buffer_for_current_stream_{0};
};

class StreamCollapseFlowUnitDataContext : public FlowUnitDataContext {
 public:
  StreamCollapseFlowUnitDataContext(Node *node, MatchKey *data_ctx_match_key,
                                    std::shared_ptr<Session> session);
  virtual ~StreamCollapseFlowUnitDataContext() = default;

  void SendEvent(std::shared_ptr<FlowUnitEvent> event) override {
    // not support user send event
  }

  void WriteInputData(std::shared_ptr<PortDataMap> stream_data_map) override;

  void CollapseNextStream();

  bool IsDataPre() override;
  bool IsDataPost() override;

  std::shared_ptr<FlowUnitInnerEvent> GenerateSendEvent() override;

  void DealWithDataPreError(std::shared_ptr<FlowUnitError> error) override;
  void DealWithProcessError(std::shared_ptr<FlowUnitError> error) override;

  void UpdateProcessState() override;

 protected:
  void UpdateInputInfo() override;

  bool SkipInheritInputToMatchNode() override { return true; };

  Status GenerateOutputPlaceholder() override;

  bool NeedStreamEndFlag() override;

  Status CheckOutputData() override;

  void UpdateBufferIndexInfo(
      std::shared_ptr<BufferIndexInfo> cur_buffer,
      std::shared_ptr<BufferIndexInfo> parent_buffer) override;

  void AppendToCache(std::shared_ptr<PortDataMap> stream_data_map);

 private:
  std::unordered_map<size_t, std::shared_ptr<PortDataMap>> stream_data_cache_;
  size_t current_collapse_order_{0};
  bool input_is_expand_from_end_buffer_{false};
  size_t output_buffer_for_current_stream_{0};
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
