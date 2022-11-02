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
  ExternalData();
  virtual ~ExternalData();
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
                   const std::shared_ptr<Stream> &init_stream);
  ~ExternalDataImpl() override;

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
  DataContext();

  virtual ~DataContext();

  virtual std::shared_ptr<BufferList> Input(const std::string &port) const = 0;

  virtual std::shared_ptr<BufferList> Output(const std::string &port) = 0;

  virtual std::shared_ptr<BufferListMap> Input() const = 0;

  virtual std::shared_ptr<BufferListMap> Output() = 0;

  virtual std::shared_ptr<BufferList> External() = 0;

  virtual std::shared_ptr<FlowUnitEvent> Event() = 0;

  virtual bool HasError() = 0;

  virtual void SendEvent(std::shared_ptr<FlowUnitEvent> event) = 0;

  virtual void SetPrivate(const std::string &key,
                          std::shared_ptr<void> private_content) = 0;

  virtual std::shared_ptr<void> GetPrivate(const std::string &key) = 0;

  virtual std::shared_ptr<DataMeta> GetInputMeta(const std::string &port) = 0;

  virtual std::shared_ptr<DataMeta> GetInputGroupMeta(
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
                      const std::shared_ptr<Session> &session);

  ~FlowUnitDataContext() override;

  std::shared_ptr<BufferList> Input(const std::string &port) const override;

  std::shared_ptr<BufferList> Output(const std::string &port) override;

  std::shared_ptr<BufferListMap> Input() const override;

  std::shared_ptr<BufferListMap> Output() override;

  std::shared_ptr<BufferList> External() override;

  void SetEvent(const std::shared_ptr<FlowUnitEvent> &event);

  std::shared_ptr<FlowUnitEvent> Event() override;

  bool HasError() override;

  void SetPrivate(const std::string &key,
                  std::shared_ptr<void> private_content) override;

  std::shared_ptr<void> GetPrivate(const std::string &key) override;

  void SendEvent(std::shared_ptr<FlowUnitEvent> event) override;

  std::shared_ptr<DataMeta> GetInputMeta(const std::string &port) override;

  std::shared_ptr<DataMeta> GetInputGroupMeta(const std::string &port) override;

  void SetOutputMeta(const std::string &port,
                     std::shared_ptr<DataMeta> data_meta) override;

  std::shared_ptr<SessionContext> GetSessionContext() override;

  std::shared_ptr<Configuration> GetSessionConfig() override;

  std::shared_ptr<StatisticsItem> GetStatistics(
      DataContextStatsType type) override;

  // common function for FlowUnitDataContext

  const std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
      &GetInputs() const;

  const std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
      &GetErrorInputs() const;

  const std::unordered_map<std::string, std::shared_ptr<BufferList>>
      &GetOutputs() const;

  const std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
      &GetExternals() const;

  void SetOutput(
      const std::unordered_map<std::string, std::shared_ptr<BufferList>>
          &data_list);

  void SetStatus(const Status &status);

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

  void SetDataPreError(bool is_error);

  /**
   * @brief call after flowunit process
   **/
  Status PostProcess();

  std::shared_ptr<Session> GetSession();

  void NotifySessionClose() override;

  // would be different in specify FlowUnitDataContext

  // buffers in stream_data_map is in order
  virtual void WriteInputData(std::shared_ptr<PortDataMap> stream_data_map);

  virtual std::shared_ptr<FlowUnitInnerEvent> GenerateSendEvent();

  virtual bool IsDataPre();

  virtual bool IsDataPost();

  virtual void DealWithDataPreError(const std::string &error_code,
                                    const std::string &error_msg);

  /**
   * @brief call after flowunit group run
   **/
  virtual void UpdateProcessState();
  virtual void ClearData();

  void Dispose();

 protected:
  virtual void UpdateBufferIndexInfo(
      const std::shared_ptr<BufferIndexInfo> &cur_buffer,
      const std::shared_ptr<BufferIndexInfo> &parent_buffer);

  virtual bool SkipInheritInputToMatchNode();

  void SetCurrentInputData(std::shared_ptr<PortDataMap> stream_data_map);

  virtual void UpdateInputInfo();

  virtual Status GenerateOutputPlaceholder();

  virtual Status GenerateOutputError();

  virtual Status GenerateOutput();

  virtual Status AppendEndFlag();

  virtual bool NeedStreamEndFlag();

  void FillPlaceholderOutput(bool from_valid_input = false,
                             bool same_with_input_num = true);

  void FillErrorOutput(bool from_valid, const std::string &error_code,
                       const std::string &error_msg,
                       bool same_with_input_num = true);

  bool HasValidOutput();

  size_t GetOutputBufferNum();

  virtual Status CheckOutputData();

  bool IsContinueProcess();

  Status process_status_{STATUS_OK};
  Status last_process_status_{STATUS_OK};

  MatchKey *data_ctx_match_key_{nullptr};
  std::shared_ptr<Session> session_;
  std::weak_ptr<SessionContext> session_context_;

  // record last input in case event sent out of Node::Run
  PortDataMap last_input_valid_data_;

  // total input
  std::shared_ptr<PortDataMap> cur_input_;
  // valid data for flowunit process
  PortDataMap cur_input_valid_data_;
  // empty for drop, empty for condition
  PortDataMap cur_input_placeholder_;
  // end for one stream, empty buffer
  PortDataMap cur_input_end_flag_;
  // error buffer
  PortDataMap cur_input_error_;
  // flowunit output
  std::unordered_map<std::string, std::shared_ptr<BufferList>>
      cur_output_valid_data_;
  // empty for drop, empty for condition
  PortDataMap cur_output_placeholder_;
  // error buffer
  PortDataMap cur_output_error_;
  // total output
  PortDataMap cur_output_;

  Node *node_;

  // state for ctx
  bool is_exception_visible_{false};
  bool is_finished_{false};  // will not process this data ctx again

  // state for stream
  bool is_empty_stream_{false};  // end_flag is first buffer of stream
  bool end_flag_received_{false};
  size_t input_stream_max_buffer_count_{0};
  size_t input_stream_cur_buffer_count_{0};
  bool end_flag_generated_{false};
  bool is_datapre_error_{false};

  // state for single run
  bool is_skippable_{false};  // no data
  std::mutex wait_user_events_lock_;
  std::unordered_set<std::shared_ptr<FlowUnitEvent>>
      wait_user_events_;  // user send event, wait to process

  bool input_has_stream_start_{false};
  bool input_has_stream_end_{false};
  bool input_valid_has_error_buffer_{false};

 private:
  void InitStatistic();

  Status UpdateOutputIndexInfo();

  std::shared_ptr<BufferProcessInfo> GetCurNodeProcessInfo(
      const std::shared_ptr<BufferIndexInfo> &index_info);

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
                            const std::shared_ptr<Session> &session);
  ~NormalFlowUnitDataContext() override;

  void SendEvent(std::shared_ptr<FlowUnitEvent> event) override;

  void UpdateProcessState() override;

 protected:
  bool NeedStreamEndFlag() override;

  void UpdateBufferIndexInfo(
      const std::shared_ptr<BufferIndexInfo> &cur_buffer,
      const std::shared_ptr<BufferIndexInfo> &parent_buffer) override;
};

class LoopNormalFlowUnitDataContext : public NormalFlowUnitDataContext {
 public:
  LoopNormalFlowUnitDataContext(Node *node, MatchKey *data_ctx_match_key,
                                const std::shared_ptr<Session> &session);
  ~LoopNormalFlowUnitDataContext() override;

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
                            const std::shared_ptr<Session> &session);
  ~StreamFlowUnitDataContext() override;

  bool IsDataPre() override;
  bool IsDataPost() override;

  void UpdateProcessState() override;

 protected:
  bool NeedStreamEndFlag() override;

  void UpdateBufferIndexInfo(
      const std::shared_ptr<BufferIndexInfo> &cur_buffer,
      const std::shared_ptr<BufferIndexInfo> &parent_buffer) override;

  PortDataMap cached_input_end_flag_;  // process after output stream end
};

class NormalExpandFlowUnitDataContext : public FlowUnitDataContext {
 public:
  NormalExpandFlowUnitDataContext(Node *node, MatchKey *data_ctx_match_key,
                                  const std::shared_ptr<Session> &session);

  ~NormalExpandFlowUnitDataContext() override;

  void UpdateProcessState() override;

 protected:
  bool NeedStreamEndFlag() override;

  void UpdateBufferIndexInfo(
      const std::shared_ptr<BufferIndexInfo> &cur_buffer,
      const std::shared_ptr<BufferIndexInfo> &parent_buffer) override;
};

class StreamExpandFlowUnitDataContext : public FlowUnitDataContext {
 public:
  StreamExpandFlowUnitDataContext(Node *node, MatchKey *data_ctx_match_key,
                                  const std::shared_ptr<Session> &session);
  ~StreamExpandFlowUnitDataContext() override;

  void WriteInputData(std::shared_ptr<PortDataMap> stream_data_map) override;

  void ExpandNextBuffer();

  bool IsDataPre() override;
  bool IsDataPost() override;

  std::shared_ptr<FlowUnitInnerEvent> GenerateSendEvent() override;

  void UpdateProcessState() override;

 protected:
  std::shared_ptr<PortDataMap> ReadFirstInCache();

  bool IsNextExpand(const std::shared_ptr<PortDataMap> &data);

  bool NeedStreamEndFlag() override;

  void UpdateBufferIndexInfo(
      const std::shared_ptr<BufferIndexInfo> &cur_buffer,
      const std::shared_ptr<BufferIndexInfo> &parent_buffer) override;

 private:
  // only read one buffer each process
  std::list<std::shared_ptr<PortDataMap>> stream_data_cache_;
  size_t cur_data_pose_in_first_cache_{0};
  size_t cur_expand_buffer_index_{0};
  bool cur_expand_buffer_index_received_{false};
  bool next_expand_buffer_event_generated_{false};
};

class NormalCollapseFlowUnitDataContext : public FlowUnitDataContext {
 public:
  NormalCollapseFlowUnitDataContext(Node *node, MatchKey *data_ctx_match_key,
                                    const std::shared_ptr<Session> &session);
  ~NormalCollapseFlowUnitDataContext() override;

  void SendEvent(std::shared_ptr<FlowUnitEvent> event) override {
    // not support user send event
  }

  bool IsDataPre() override;
  bool IsDataPost() override;

  void UpdateProcessState() override;

  Status GenerateOutputError() override;

 protected:
  bool SkipInheritInputToMatchNode() override { return true; };

  Status GenerateOutputPlaceholder() override;

  bool NeedStreamEndFlag() override;

  Status CheckOutputData() override;

  Status GenerateOutput() override;

  void UpdateBufferIndexInfo(
      const std::shared_ptr<BufferIndexInfo> &cur_buffer,
      const std::shared_ptr<BufferIndexInfo> &parent_buffer) override;

 private:
  size_t output_buffer_for_current_stream_{0};
};

class StreamCollapseFlowUnitDataContext : public FlowUnitDataContext {
 public:
  StreamCollapseFlowUnitDataContext(Node *node, MatchKey *data_ctx_match_key,
                                    const std::shared_ptr<Session> &session);
  ~StreamCollapseFlowUnitDataContext() override;

  void SendEvent(std::shared_ptr<FlowUnitEvent> event) override;

  void WriteInputData(std::shared_ptr<PortDataMap> stream_data_map) override;

  void CollapseNextStream();

  bool IsDataPre() override;
  bool IsDataPost() override;

  std::shared_ptr<FlowUnitInnerEvent> GenerateSendEvent() override;

  void UpdateProcessState() override;

  Status GenerateOutputError() override;

 protected:
  void UpdateInputInfo() override;

  bool SkipInheritInputToMatchNode() override { return true; };

  Status GenerateOutputPlaceholder() override;

  bool NeedStreamEndFlag() override;

  Status CheckOutputData() override;

  Status GenerateOutput() override;

  void UpdateBufferIndexInfo(
      const std::shared_ptr<BufferIndexInfo> &cur_buffer,
      const std::shared_ptr<BufferIndexInfo> &parent_buffer) override;

  void AppendToCache(const std::shared_ptr<PortDataMap> &stream_data_map);

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
  ~ExecutorDataContext() override;

  std::shared_ptr<BufferList> Input(const std::string &port) const override;

  std::shared_ptr<BufferList> Output(const std::string &port) override;

  std::shared_ptr<BufferListMap> Input() const override;

  std::shared_ptr<BufferListMap> Output() override;

  std::shared_ptr<BufferList> External() override;

  bool HasError() override;

  std::shared_ptr<FlowUnitEvent> Event() override;

  void SendEvent(std::shared_ptr<FlowUnitEvent> event) override;

  void SetPrivate(const std::string &key,
                  std::shared_ptr<void> private_content) override;

  std::shared_ptr<void> GetPrivate(const std::string &key) override;

  std::shared_ptr<DataMeta> GetInputMeta(const std::string &port) override;

  std::shared_ptr<DataMeta> GetInputGroupMeta(const std::string &port) override;

  void SetOutputMeta(const std::string &port,
                     std::shared_ptr<DataMeta> data_meta) override;

  std::shared_ptr<SessionContext> GetSessionContext() override;

  void SetStatus(const Status &status);

  std::shared_ptr<Configuration> GetSessionConfig() override;

  std::shared_ptr<StatisticsItem> GetStatistics(
      DataContextStatsType type) override;

  void Clear();

 private:
  std::shared_ptr<FlowUnitDataContext> origin_ctx_;
  std::shared_ptr<FlowUnitExecData> data_;
};

}  // namespace modelbox
#endif
