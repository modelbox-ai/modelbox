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

#ifndef DATA_HANDLER_H_
#define DATA_HANDLER_H_
#include "buffer.h"
#include "buffer_list.h"
#include "graph.h"
#include "modelbox/context.h"
#include "node.h"

namespace modelbox {

#define DEFAULT_PORT_NAME "__default__"

class ModelBoxEngine;

/**
 * @brief data handler used by  dynamic graph,can store data and node
 * information
 */
class DataHandler : public std::enable_shared_from_this<DataHandler> {
 public:
  DataHandler(BindNodeType type = BUFFERLIST_NODE,
              const std::shared_ptr<ModelBoxEngine> &env = nullptr);

  virtual ~DataHandler();

  /**
   * @brief close data handler, when data handler is closd,no data can be pushed
   */
  void Close();
  /**
   * @brief  push data to data handler of the node
   * @param key port  name of the node
   * @param data data handler which stores data
   * @return  return result
   */
  Status PushData(std::shared_ptr<DataHandler> &data, const std::string &key);

  Status PushData(std::shared_ptr<Buffer> &data, const std::string &key);

  Status PushData(std::shared_ptr<BufferList> &data, const std::string &key);
  /**
   * @brief push meta to data handler
   * @param key  key of meta info
   * @param data  data of meta info
   * @return return result
   */
  Status SetMeta(const std::string &key, const std::string &data);

  std::shared_ptr<DataHandler> operator[](const std::string &port_name);

  /**
   * @brief get data from data handler via key
   * @param key: port name
   * @return  data handler contains data of port(key)
   */
  std::shared_ptr<DataHandler> GetDataHandler(const std::string &key);

  Status SetDataHandler(
      const std::map<std::string, std::shared_ptr<DataHandler>> &data_map);
  /**
   * @brief get bufferlist via key
   * @param key: port name
   * @return bufferlist on port(key)
   */
  std::shared_ptr<BufferList> GetBufferList(const std::string &key);

  /**
   * @brief get data on outports one by one
   * @return data hadnler store output data
   */
  std::shared_ptr<DataHandler> GetData();

  std::string GetMeta(std::string &key);
  /**
   * @brief get flow error
   * @return error code
   */
  Status GetError();

  // for output: record the node name
  void SetNodeName(const std::string &name);

  std::string GetNodeName();

  void SetError(const Status &status);

 private:
  /*
   check input stream has been closed
  */
  bool IsClosed();
  void SetEnv(const std::shared_ptr<ModelBoxEngine> &env);
  std::shared_ptr<ModelBoxEngine> GetEnv();
  Status InsertOutputNode(std::shared_ptr<HandlerContext> &context);
  /*
  bind gcgraph for datahandler
  */
  Status SetBindGraph(const std::shared_ptr<GraphState> &gcgraph);
  /*
  get bind graph
  */
  std::shared_ptr<GraphState> GetBindGraph();
  /*
  for output: record the node type
  */
  DataHandlerType GetDataHandlerType();
  void SetDataHandlerType(const DataHandlerType &type);

  /* for input: when node has one more port, check whether the input
   datahanlders is same nodetype or not
   */
  Status CheckInputType(BindNodeType &node_type);

  /*
  for output: save outport names
  */
  std::set<std::string> GetPortNames();
  Status SetPortNames(std::set<std::string> &port_names);

  /*
  get and set bind node type: stream or normal
  */
  BindNodeType GetBindNodeType();
  void SetBindNodeType(BindNodeType type);

  /*
  set extern  map and bufferlist, used when feed data
  */
  void SetExternData(std::shared_ptr<void> extern_map,
                     std::shared_ptr<BufferList> &bufferlist);

  /*
 for input: get inport-outport map when datahandler is constructed with one more
 datahandler
 */
  std::unordered_map<std::string, std::string> GetPortMap();

  std::shared_ptr<DataHandler> GetOutputData(
      std::shared_ptr<DynamicGraph> &dynamic_graph);

  friend class SingleNode;
  friend class ModelBoxEngine;
  friend class InputContext;
  friend class StreamContext;
  bool closed_{false};

  std::weak_ptr<ModelBoxEngine> env_;

  Status error_{STATUS_SUCCESS};
  DataHandlerType data_handler_type_{INPUT};
  BindNodeType data_type_{BUFFERLIST_NODE};

  std::string node_name_;
  std::set<std::string> port_names_;
  std::unordered_map<std::string, std::string> port_to_port_;
  std::unordered_map<std::string, std::string> port_to_node_;
  std::unordered_map<std::string, BindNodeType> node_type_map_;

  std::shared_ptr<HandlerContext> context_;
};

}  // namespace modelbox
#endif