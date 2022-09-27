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

#ifndef MODELBOX_FLOW_H_
#define MODELBOX_FLOW_H_

#include <modelbox/base/graph_manager.h>
#include <modelbox/base/log.h>
#include <modelbox/base/status.h>
#include <modelbox/flow_graph_desc.h>
#include <modelbox/flow_stream_io.h>
#include <modelbox/flowunit.h>
#include <modelbox/flowunit_builder.h>
#include <modelbox/graph.h>
#include <modelbox/profiler.h>

#include <memory>
#include <string>

namespace modelbox {

constexpr const char* DEFAULT_FLOW_PATH =
    "/usr/local/share/modelbox/solutions/graphs";

/**
 * @brief modelbox flow control
 */
class Flow {
 public:
  /**
   * @brief Graph format
   */
  enum Format {
    FORMAT_AUTO,
    FORMAT_TOML,
    FORMAT_JSON,
    FORMAT_UNKNOWN,
  };

  Flow();
  virtual ~Flow();

  /**
   * @brief register flow unit to flow
   * @param flowunit_builder flow unit builder
   * @return
   **/
  void RegisterFlowUnit(
      const std::shared_ptr<modelbox::FlowUnitBuilder>& flowunit_builder);

  /**
   * @brief Init flow from file
   * @param configfile path to config file, support toml and json
   * @param format config file format, when auto, Flow will guess format.
   * @return init result.
   */
  Status Init(const std::string& configfile, Format format = FORMAT_AUTO);

  /**
   * @brief Init flow from inline graph
   * @param name graph name
   * @param graph inline graph.
   * @param format config file format, when auto, Flow will guess format.
   * @return init result.
   */
  Status Init(const std::string& name, const std::string& graph,
              Format format = FORMAT_AUTO);

  /**
   * @brief Init flow from input stream
   * @param is input stream of graph.
   * @param fname graph name
   * @return init result.
   */
  Status Init(std::istream& is, const std::string& fname);

  /**
   * @brief Init flow from configuration
   * @param config configuration object
   * @return init result.
   */
  Status Init(std::shared_ptr<Configuration> config);

  /**
   * @brief  init flow from FlowGraphDesc
   * @param flow_graph_desc  graph desc
   * @return init result.
   */
  Status Init(const std::shared_ptr<FlowGraphDesc>& flow_graph_desc);

  /**
   * @brief init flow by name, args and flow directory
   * @param name flow name
   * @param args flow args
   * @param flow_dir scan flow directory
   * @return init result.
   */
  Status InitByName(
      const std::string& name,
      const std::unordered_map<std::string, std::string>& args = {},
      const std::string& flow_dir = DEFAULT_FLOW_PATH);

  /**
   * @brief return until flow running
   * @return start result.
   */
  Status StartRun();

  /**
   * @brief Build graph
   * @return build result.
   */
  Status Build();

  /**
   * @brief Run graph, block until graph is finish.
   * @return run result.
   */
  Status Run();

  /**
   * @brief Run graph async.
   * @return run result.
   */
  Status RunAsync();

  /**
   * @brief Wait graph run finish
   * @param millisecond wait timeout
   * @param ret_val graph run result
   * @return wait result.
   */
  Status Wait(int64_t millisecond = 0, Status* ret_val = nullptr);

  /**
   * @brief Force stop graph
   */
  void Stop();

  /**
   * @brief Create external data
   * @return extern data
   */
  std::shared_ptr<ExternalDataMap> CreateExternalDataMap();

  /**
   * @brief Create stream io to send and recv stream data
   * @return FlowStreamIO
   */
  std::shared_ptr<FlowStreamIO> CreateStreamIO();

  /**
   * @brief Get profiler
   * @return profiler
   */
  std::shared_ptr<Profiler> GetProfiler();

  /**
   * @brief Get graph id
   * @return graph id
   */
  std::string GetGraphId() const;

  /**
   * @brief Get graph name
   * @return graph name
   */
  std::string GetGraphName() const;

 private:
  Status InitComponent();

  void Clear();

  Status ConfigFileRead(const std::string& configfile, Format format,
                        std::istringstream* ifs);

  Status GetConfigByGraphFile(const std::string& configfile,
                              std::shared_ptr<Configuration>& config,
                              Format format);

  Status GetGraphFilePathByName(const std::string& flow_name,
                                const std::string& graph_dir,
                                std::string& graph_path);

  Status GetInputArgs(
      std::shared_ptr<Configuration>& config,
      const std::unordered_map<std::string, std::string>& input_args = {});

  Status GuessConfFormat(const std::string& configfile, const std::string& data,
                         enum Format* format);

  std::list<std::shared_ptr<FlowUnitFactory>> flowunit_factory_list_;

  std::shared_ptr<Drivers> drivers_;
  std::shared_ptr<DeviceManager> device_mgr_;
  std::shared_ptr<FlowUnitManager> flowunit_mgr_;
  std::shared_ptr<GraphConfigManager> graphconf_mgr_;
  std::shared_ptr<Configuration> config_;
  std::shared_ptr<GraphConfig> graphconfig_;
  std::shared_ptr<GCGraph> gcgraph_;
  std::shared_ptr<Graph> graph_;
  std::shared_ptr<Profiler> profiler_;
  bool timer_run_ = false;
  std::shared_ptr<std::unordered_map<std::string, std::string>> args_;
};

}  // namespace modelbox
#endif  // MODELBOX_FLOW_H_
