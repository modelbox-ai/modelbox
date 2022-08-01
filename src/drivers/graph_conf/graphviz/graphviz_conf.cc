/*
 *  * Copyright (C) 2020 Huawei Technologies Co., Ltd. All rights reserved.
 */

#include "graphviz_conf.h"

#include "modelbox/base/status.h"

namespace modelbox {

thread_local std::stringstream g_graphviz_error;

int GraphvizeError(char *errmsg) {
  // errmsg might be part for one specify error
  g_graphviz_error << errmsg;
  return 0;
}

std::mutex kCgraphLock;

GraphvizFactory::GraphvizFactory() {
  std::unique_lock<std::mutex> lock(kCgraphLock);
  agseterrf(GraphvizeError);
}

GraphvizFactory::~GraphvizFactory() {
  std::unique_lock<std::mutex> lock(kCgraphLock);
  agseterrf(nullptr);
}

std::shared_ptr<GraphConfig> GraphvizFactory::CreateGraphConfigFromStr(
    const std::string &graph_config) {
  std::shared_ptr<modelbox::GraphConfig> graphviz_conf =
      std::make_shared<modelbox::GraphvizConfig>(graph_config, false);
  return graphviz_conf;
}

std::shared_ptr<GraphConfig> GraphvizFactory::CreateGraphConfigFromFile(
    const std::string &file_path) {
  std::shared_ptr<modelbox::GraphConfig> graphviz_conf =
      std::make_shared<modelbox::GraphvizConfig>(file_path, true);
  return graphviz_conf;
}

std::string GraphvizFactory::GetGraphConfFactoryType() { return factory_type_; }

GraphvizConfig::GraphvizConfig(const std::string &graph_conf,
                               const bool is_file) {
  graphviz_conf_ = graph_conf;
  is_file_ = is_file;
}

GraphvizConfig::~GraphvizConfig() = default;

std::shared_ptr<GCGraph> GraphvizConfig::Resolve() {
  std::shared_ptr<modelbox::GCGraph> graph = std::make_shared<modelbox::GCGraph>();
  std::shared_ptr<Agraph_t> g;

  auto init_status = graph->Init(graph);
  if (init_status != STATUS_OK) {
    MBLOG_ERROR << "gcgraph init failed";
    return nullptr;
  }

  if (is_file_) {
    g = LoadGraphFromFile();
  } else {
    g = LoadGraphFromStr();
  }

  if (g == nullptr) {
    MBLOG_ERROR << "load graph faild.";
    return nullptr;
  }

  auto *graph_name = agnameof(g.get());
  if (graph_name != nullptr) {
    graph->SetGraphName(graph_name);
  }

  auto ret = TraversalsGraph(g, graph);
  if (!ret) {
    MBLOG_ERROR << "traversals graph faild.";
    return nullptr;
  }

  ret = TraversalsNode(g, graph);
  if (!ret) {
    MBLOG_ERROR << "traversals node faild.";
    return nullptr;
  }

  ret = TraversalsEdge(g, graph);
  if (!ret) {
    MBLOG_ERROR << "traversals edge faild.";
    return nullptr;
  }

  return graph;
}

std::shared_ptr<Agraph_t> GraphvizConfig::LoadGraphFromStr() {
  Agraph_t *g = nullptr;

  std::unique_lock<std::mutex> lock(kCgraphLock);
  g_graphviz_error.clear();
  g = agmemread(graphviz_conf_.c_str());

  if (g == nullptr) {
    MBLOG_ERROR << "load graph from str failed, graphviz config is : "
                << std::endl
                << graphviz_conf_;
    MBLOG_ERROR << "graphviz: " << g_graphviz_error.str();
    g_graphviz_error.clear();
    StatusError = {STATUS_BADCONF};
    return nullptr;
  }

  std::shared_ptr<Agraph_t> ret(g, [](Agraph_t *ptr) {
    std::unique_lock<std::mutex> lock(kCgraphLock);
    agclose(ptr);
  });
  return ret;
}

std::shared_ptr<Agraph_t> GraphvizConfig::LoadGraphFromFile() {
  Agraph_t *g = nullptr;
  std::string file = PathCanonicalize(graphviz_conf_);
  if (file.length() == 0) {
    MBLOG_ERROR << "graph path is invalid, " << file;
    StatusError = {STATUS_BADCONF, "path is invalid"};
    return nullptr;
  }

  FILE *fp = fopen(file.c_str(), "r");
  if (fp == nullptr) {
    MBLOG_ERROR << "open file failed, file: " << file << ", "
                << StrError(errno);
    StatusError = {STATUS_BADCONF, StrError(errno)};
    return nullptr;
  }

  std::unique_lock<std::mutex> lock(kCgraphLock);
  g = agread(fp, nullptr);
  fclose(fp);
  fp = nullptr;

  if (g == nullptr) {
    std::string errmsg = "read graph failed. ";
    if (aglasterr()) {
      errmsg += aglasterr();
    }
    StatusError = {STATUS_BADCONF, errmsg};
    return nullptr;
  }

  std::shared_ptr<Agraph_t> ret(g, [](Agraph_t *ptr) {
    std::unique_lock<std::mutex> lock(kCgraphLock);
    agclose(ptr);
  });

  return ret;
}

Status GraphvizConfig::TraversalsGraph(std::shared_ptr<Agraph_t> g,
                                       std::shared_ptr<GCGraph> graph) {
  std::vector<std::string> node_keys;
  Agsym_t *sym = nullptr;

  if (g == nullptr || graph == nullptr) {
    MBLOG_ERROR << "graph is null.";
    return STATUS_INVALID;
  }

  while (true) {
    sym = agnxtattr(g.get(), AGRAPH, sym);
    if (sym == nullptr) {
      break;
    }

    node_keys.emplace_back(sym->name);
  }

  for (const std::string &elem : node_keys) {
    auto *agget_str = agget(g.get(), const_cast<char *>(elem.c_str()));
    if (agget_str == nullptr) {
      MBLOG_ERROR << "failed to get graph attr name: " << elem;
      continue;
    }

    graph->SetConfiguration(elem, agget_str);
  }

  return STATUS_OK;
}

Status GraphvizConfig::TraversalsNode(std::shared_ptr<Agraph_t> g,
                                      std::shared_ptr<GCGraph> graph) {
  std::vector<std::string> node_keys;
  Agnode_t *agnode = nullptr;

  Agsym_t *sym = nullptr;
  while (true) {
    sym = agnxtattr(g.get(), AGNODE, sym);
    if (sym == nullptr) {
      break;
    }
    node_keys.emplace_back(sym->name);
  }

  for (agnode = agfstnode(g.get()); agnode;
       agnode = agnxtnode(g.get(), agnode)) {
    auto gcnode = std::make_shared<GCNode>();
    auto *agname = agnameof(agnode);
    if (agname == nullptr) {
      return {STATUS_BADCONF, "agname is invalid"};
    }

    auto status = gcnode->Init(agname, graph);
    if (!status) {
      return status;
    }

    MBLOG_DEBUG << "add node: " << gcnode->GetNodeName();
    for (const std::string &elem : node_keys) {
      const char *ag_value = agget(agnode, const_cast<char *>(elem.c_str()));
      if (ag_value == nullptr) {
        continue;
      }

      std::string value = ag_value;
      if (value == "") {
        continue;
      }

      gcnode->SetConfiguration(elem, value);
      MBLOG_DEBUG << "  key: " << elem << ", value: " << value;
    }

    graph->AddNode(gcnode);
  }

  return STATUS_OK;
}

std::shared_ptr<modelbox::GCEdge> GraphvizConfig::NewGcEdgeFromAgedge(
    std::shared_ptr<GCGraph> graph, Agedge_t *agedge) {
  auto gcedge = std::make_shared<modelbox::GCEdge>();
  auto ret = gcedge->Init(graph);
  if (!ret) {
    StatusError = ret;
    return nullptr;
  }

  /* IMPORTANT NOTES:
   * The HEAD and TAIL of an edge is totally inverse between Graphviz and
   * GCGraph. 
   * For Graphviz, an edge arrow is pointing to the HEAD node from the
   * TAIL node:
   *
   *                           Graphviz_Edge
   *           Graphviz_TAIL   ------------>   Graphviz_HEAD
   *
   * For GCGraph, inversely, an edge arrow is pointing to the TAIL node
   * (destination node) from the HEAD node (source node):
   *
   *                           GCGraph_Edge
   *           GCGraph_HEAD    ------------>   GCGraph_TAIL
   */

  std::string head_node_name;
  auto *node_name = agnameof(agtail(agedge));
  if (node_name == nullptr) {
    head_node_name = "";
  } else {
    head_node_name = node_name;
  }

  auto head_node = graph->GetNode(head_node_name);
  if (head_node == nullptr) {
    MBLOG_ERROR << "head node [" << head_node_name << "]"
                << " not exist.";
    StatusError = {STATUS_FAULT, "get head node failed."};
    return nullptr;
  }

  gcedge->SetHeadNode(head_node);
  std::string tail_node_name;
  node_name = agnameof(aghead(agedge));
  if (node_name == nullptr) {
    tail_node_name = "";
  } else {
    tail_node_name = node_name;
  }

  auto tail_node = graph->GetNode(tail_node_name);
  if (tail_node == nullptr) {
    MBLOG_ERROR << "tail node [" << tail_node_name << "]"
                << " not exist.";
    StatusError = {STATUS_FAULT, "get tail node failed."};
    return nullptr;
  }

  gcedge->SetTailNode(tail_node);

  return gcedge;
}

Status GraphvizConfig::TraversalsEdge(std::shared_ptr<Agraph_t> g,
                                      std::shared_ptr<GCGraph> graph) {
  Agnode_t *agnode = nullptr;
  Agedge_t *agedge = nullptr;
  std::vector<std::string> edge_keys;

  Agsym_t *sym = nullptr;
  while (true) {
    sym = agnxtattr(g.get(), AGEDGE, sym);
    if (sym == nullptr) {
      break;
    }
    edge_keys.emplace_back(sym->name);
  }

  for (agnode = agfstnode(g.get()); agnode;
       agnode = agnxtnode(g.get(), agnode)) {
    for (agedge = agfstout(g.get(), agnode); agedge;
         agedge = agnxtout(g.get(), agedge)) {
      auto gcedge = NewGcEdgeFromAgedge(graph, agedge);
      if (gcedge == nullptr) {
        return {StatusError};
      }

      for (const std::string &elem : edge_keys) {
        char *value = nullptr;
        value = agget(agedge, const_cast<char *>(elem.c_str()));
        if (value == nullptr || *value == '\0') {
          continue;
        }
        gcedge->SetConfiguration(elem, value);
        if (elem == GRAPHVIZ_HEAD_PORT) {
          gcedge->SetTailPort(value);
          gcedge->GetTailNode()->SetInputPort(value);
        } else if (elem == GRAPHVIZ_TAIL_PORT) {
          gcedge->SetHeadPort(value);
          gcedge->GetHeadNode()->SetOutputPort(value);
        }
      }

      auto ret = graph->AddEdge(gcedge);
      if (!ret) {
        return ret;
      }
    }
  }

  return STATUS_OK;
}

}  // namespace modelbox