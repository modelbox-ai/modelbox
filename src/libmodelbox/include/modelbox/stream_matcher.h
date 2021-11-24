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

#ifndef MODELBOX_STREAM_MATCHER_H_
#define MODELBOX_STREAM_MATCHER_H_

#include <map>
#include <memory>
#include <vector>

#include "modelbox/port.h"

namespace modelbox {

class GraphStream : public std::enable_shared_from_this<GraphStream> {
  // we can only have one relation stream parent or full
 public:
  GraphStream();
  virtual ~GraphStream() = default;
  std::shared_ptr<GraphStream> GenerateConditionStream(
      std::shared_ptr<GraphStream> other_stream);
  std::shared_ptr<GraphStream> GenerateChildStream();
  std::shared_ptr<GraphStream> GetParentStream();
  std::shared_ptr<GraphStream> GetFullStream();
  std::shared_ptr<GraphStream> GenerateSiblingStream();

  bool IsSameConditonGroupStream(std::shared_ptr<GraphStream> other_stream);

 private:
  std::shared_ptr<GraphStream> parent_stream_;
  std::shared_ptr<GraphStream> full_stream_;
  // other conditon stream
  std::weak_ptr<GraphStream> next_stream_;
};

class StreamMatcher {
 public:
  StreamMatcher(std::set<std::shared_ptr<NodeBase>> start_nodes,
                std::set<std::shared_ptr<NodeBase>> all_nodes);
  virtual ~StreamMatcher();
  Status StartCheck();
  std::shared_ptr<GraphStream> GenerateRootStream();

 private:
  std::shared_ptr<GraphStream> GetInputStream(std::shared_ptr<NodeBase> node);

  Status GenerateOutputStream(std::shared_ptr<NodeBase> node,
                              std::shared_ptr<GraphStream> input_stream);

  Status BindOutputStream(std::shared_ptr<NodeBase> node,
                          std::shared_ptr<OutPort> output_port,
                          std::shared_ptr<GraphStream> output_stream);

  Status UpdateCheckingNode();
  std::shared_ptr<GraphStream> root_stream_;
  std::map<std::shared_ptr<InPort>, std::shared_ptr<GraphStream>>
      port_stream_map_;
  std::map<std::shared_ptr<InPort>, std::vector<std::shared_ptr<OutPort>>>
      port_connect_map_;
  std::set<std::shared_ptr<NodeBase>> checked_nodes_;
  std::set<std::shared_ptr<NodeBase>> checking_nodes_;
  std::set<std::shared_ptr<NodeBase>> all_nodes_;
};

}  // namespace modelbox
#endif  // MODELBOX_STREAM_CHECK_H_