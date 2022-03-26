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
#include "node.h"

namespace modelbox {

enum DataHandlerType { INPUT = 0, OUTPUT };
class ModelBoxEngine;

class DataHandler : public std::enable_shared_from_this<DataHandler> {
 public:
  DataHandler(const std::shared_ptr<ModelBoxEngine> &env = nullptr);

  ~DataHandler();

  std::shared_ptr<DataHandler> operator[](const std::string &port_name);

  /**
   * @brief get data from data handler via port_name
   * @param key: port name
   * @return  data handler contains data of port(key)
   */
  std::shared_ptr<DataHandler> GetDataHandler(const std::string &key);
  /**
   * @brief: Combind a datahandler with some datahandlers bind port
   * @param data_map: datahandlers related specify port
   */
  Status SetDataHandler(
      const std::map<std::string, std::shared_ptr<DataHandler>> &data_map);

  /**
   * @brief get flow error
   * @return error code
   */
  Status GetError() { return error_; }

  /**
   * @brief set flowunit name for datahandler
   * @param name: flowunit name
   */
  void SetNodeName(const std::string &name);
  /**
   * @brief get flowunit name of datahandler
   * @return flowunit name
   */
  std::string GetNodeName();
  /**
   * @brief save error status of modelbox
   * @param status: Status of modelbox
   */
  void SetError(const Status &status) { error_ = status; }

 private:
  void SetEnv(const std::shared_ptr<ModelBoxEngine> &env);
  std::shared_ptr<ModelBoxEngine> GetEnv();

  DataHandlerType GetDataHandlerType();
  void SetDataHandlerType(const DataHandlerType &type);
  std::set<std::string> GetPortNames();
  Status SetPortNames(std::set<std::string> &outport_names);
  std::unordered_map<std::string, std::string> GetPortMap() {
    return port_to_port_;
  };

 private:
  friend class ModelBoxEngine;
  Status error_{STATUS_SUCCESS};
  DataHandlerType data_handler_type_{INPUT};
  std::string node_name_{""};
  std::set<std::string> port_names_;
  std::unordered_map<std::string, std::string> port_to_port_;
  std::unordered_map<std::string, std::string> port_to_node_;
  std::weak_ptr<ModelBoxEngine> engine_;
};

};  // namespace modelbox
#endif