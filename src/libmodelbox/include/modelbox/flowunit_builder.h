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

#ifndef MODELBOX_FLOW_UNIT_BUILDER_H_
#define MODELBOX_FLOW_UNIT_BUILDER_H_

#include <memory>

#include "flowunit.h"

namespace modelbox {
class FlowUnitBuilder {
 public:
  virtual void Probe(std::shared_ptr<FlowUnitDesc> &desc) = 0;

  virtual std::shared_ptr<FlowUnit> Build() = 0;
};

class RegFlowUnitFactory : public FlowUnitFactory {
 public:
  RegFlowUnitFactory(std::shared_ptr<FlowUnitBuilder> builder);

  std::map<std::string, std::shared_ptr<FlowUnitDesc>> FlowUnitProbe() override;

  std::string GetFlowUnitFactoryType() override;

  std::string GetFlowUnitFactoryName() override;

  std::shared_ptr<FlowUnit> CreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type) override;

 private:
  std::shared_ptr<FlowUnitBuilder> builder_;
  std::string unit_type_;
  std::string unit_name_;
};

}  // namespace modelbox

#endif  // MODELBOX_FLOW_UNIT_BUILDER_H_