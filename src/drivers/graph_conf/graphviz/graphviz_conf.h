/*
 *  * Copyright (C) 2020 Huawei Technologies Co., Ltd. All rights reserved.
 */

#include <modelbox/base/graph_manager.h>
#include <graphviz/cgraph.h>
#include <stdio.h>

#include <iostream>

#ifndef MODELBOX_GRAPHVIZ_CONF_H
#define MODELBOX_GRAPHVIZ_CONF_H

namespace modelbox {

constexpr const char *GRAPHCONF_TYPE = "graph";
constexpr const char *GRAPHCONF_NAME = "graphconf-graphvize";
constexpr const char *GRAPHCONF_DESC = "graph config parse graphviz";
constexpr const char *GRAPHVIZE_VERSION = "1.0.0";

constexpr const char *GRAPHVIZ_HEAD_PORT = "headport";
constexpr const char *GRAPHVIZ_TAIL_PORT = "tailport";
constexpr const char *FACTORY_TYPE_GRAPHVIZE = "graphviz";

class GraphvizFactory : public GraphConfigFactory {
 private:
  std::string factory_type_ = FACTORY_TYPE_GRAPHVIZE;

 public:
  GraphvizFactory();

  ~GraphvizFactory() override;

  std::shared_ptr<GraphConfig> CreateGraphConfigFromStr(
      const std::string &graph_config) override;

  std::shared_ptr<GraphConfig> CreateGraphConfigFromFile(
      const std::string &file_path) override;

  std::string GetGraphConfFactoryType() override;
};

class GraphvizConfig : public GraphConfig {
 public:
  GraphvizConfig(const std::string &graph_conf, bool is_file);

  ~GraphvizConfig() override;

  std::shared_ptr<GCGraph> Resolve() override;

 private:
  std::shared_ptr<Agraph_t> LoadGraphFromStr();

  std::shared_ptr<Agraph_t> LoadGraphFromFile();

  Status TraversalsGraph(const std::shared_ptr<Agraph_t> &g,
                         const std::shared_ptr<GCGraph> &graph);

  Status TraversalsNode(const std::shared_ptr<Agraph_t> &g,
                        const std::shared_ptr<GCGraph> &graph);

  std::shared_ptr<modelbox::GCEdge> NewGcEdgeFromAgedge(
      const std::shared_ptr<GCGraph> &graph, Agedge_t *agedge);

  Status TraversalsEdge(const std::shared_ptr<Agraph_t> &g,
                        const std::shared_ptr<GCGraph> &graph);

  Status TraversalsSubGraph(std::shared_ptr<Agraph_t> g,
                            std::shared_ptr<GCGraph> graph);

  std::string graphviz_conf_;
  bool is_file_;
  std::mutex lock_;
};

}  // namespace modelbox

#endif  // MODELBOX_GRAPHVIZ_CONF_H
