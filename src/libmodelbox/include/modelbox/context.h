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

#ifndef HANDLER_CONTEXT_
#define HANDLER_CONTEXT_

#include "modelbox/buffer.h"
#include "modelbox/buffer_list.h"
#include "modelbox/graph.h"
#include "modelbox/node.h"

namespace modelbox {
typedef std::unordered_map<std::string, std::shared_ptr<Buffer>> BufferMap;
enum DataHandlerType { INPUT = 0, OUTPUT };
enum BindNodeType { STREAM_NODE = 0, BUFFERLIST_NODE = 1, VIRTUAL_NODE = 2 };
class DataHandler;
class ModelBoxEngine;

class GraphState {
 public:
  std::shared_ptr<GCGraph> gcgraph_;
  std::shared_ptr<DynamicGraph> graph_;
  std::shared_ptr<void> external_data_;
  Status error_{STATUS_SUCCESS};
};
class HandlerContext {
 public:
  friend class ModelBoxEngine;
  HandlerContext(std::weak_ptr<ModelBoxEngine> &env);
  virtual ~HandlerContext();

  virtual Status PushData(const std::string &key,
                          const std::shared_ptr<BufferList> &bufferlist) = 0;

  virtual std::shared_ptr<BufferList> GetBufferList(const std::string &key) = 0;

  virtual Status RunGraph(const std::shared_ptr<DataHandler> &) = 0;

  void SetMeta(const std::string &key, const std::string &value);

  std::string GetMeta(const std::string &key);

  std::shared_ptr<GraphState> GetGraphState();

  void SetGraphState(const std::shared_ptr<GraphState> & /*state*/);

  virtual void Close();

  void SetFlowUnitDesc(const std::shared_ptr<FlowUnitDesc> &desc);

  std::shared_ptr<FlowUnitDesc> GetFlowUnitDesc();

  std::shared_ptr<FlowUnitDesc> desc_;
  std::unordered_map<std::string, std::shared_ptr<BufferList>> data_map_;
  std::weak_ptr<ModelBoxEngine> env_;

 private:
  std::unordered_map<std::string, std::string> meta_;
  std::shared_ptr<GraphState> graph_state_;
};

class InputContext : public HandlerContext {
 public:
  InputContext(std::weak_ptr<ModelBoxEngine> env);
  ~InputContext() override;
  void SetExternPtr(std::shared_ptr<void> extern_data_map,
                    std::shared_ptr<BufferList> extern_buffer_list);
  Status PushData(const std::string &key,
                  const std::shared_ptr<BufferList> &bufferlist) override;

  std::shared_ptr<BufferList> GetBufferList(const std::string &key) override;

  Status RunGraph(const std::shared_ptr<DataHandler> &handler) override;
  void Close() override;

 private:
  std::weak_ptr<ModelBoxEngine> env_;
  std::shared_ptr<void> extern_data_map_;
  std::shared_ptr<BufferList> extern_buffer_list_;
};

class StreamContext : public HandlerContext {
 public:
  StreamContext(std::weak_ptr<ModelBoxEngine> env);
  ~StreamContext() override;
  Status PushData(const std::string &key,
                  const std::shared_ptr<BufferList> &bufferlist) override;

  std::shared_ptr<BufferList> GetBufferList(const std::string &key) override;

  Status RunGraph(const std::shared_ptr<DataHandler> & /*handler*/) override;

 private:
  bool end_flag_;
};

class BufferListContext : public HandlerContext {
 public:
  BufferListContext(std::weak_ptr<ModelBoxEngine> env);
  ~BufferListContext() override;
  Status PushData(const std::string &key,
                  const std::shared_ptr<BufferList> &bufferlist) override;

  std::shared_ptr<BufferList> GetBufferList(const std::string &key) override;

  Status RunGraph(const std::shared_ptr<DataHandler> & /*handler*/) override;
};

}  // namespace modelbox
#endif