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

#include "flowunit_mockflowunit.h"

#include <utility>

namespace modelbox {

MockDriverFlowUnit MockDriverFlowUnit::desc_;

std::map<std::string, std::shared_ptr<FlowUnitDesc>>
MockFlowUnitFactory::FlowUnitProbe() {
  auto tmp_map = std::map<std::string, std::shared_ptr<FlowUnitDesc>>();
  if (flowunit_desc_.size() > 0) {
    for (auto &desc : flowunit_desc_) {
      tmp_map.insert(std::make_pair(desc->GetFlowUnitName(), desc));
    }
  }

  if (bind_mock_flowunit_ != nullptr) {
    auto desc = bind_mock_flowunit_->GetFlowUnitDesc();
    tmp_map.insert(std::make_pair(
        bind_mock_flowunit_->GetFlowUnitDesc()->GetFlowUnitName(), desc));
  }

  return tmp_map;
}

std::shared_ptr<FlowUnit> MockFlowUnitFactory::CreateFlowUnit(
    const std::string &name, const std::string &type) {
  if (flowunit_create_func_) {
    return flowunit_create_func_(name, type);
  }

  if (bind_mock_flowunit_ != nullptr) {
    return bind_mock_flowunit_;
  }

  return std::make_shared<MockFlowUnit>();
}

void MockFlowUnitFactory::SetMockFunctionFlowUnit(
    std::shared_ptr<MockFlowUnit> mock_flowunit) {
  bind_mock_flowunit_ = std::move(mock_flowunit);
}

void MockFlowUnitFactory::SetMockCreateFlowUnitFunc(
    std::function<std::shared_ptr<FlowUnit>(const std::string &name,
                                            const std::string &type)>
        create_func) {
  flowunit_create_func_ = std::move(create_func);
}

void MockFlowUnitFactory::SetMockFlowUnitDesc(
    std::vector<std::shared_ptr<FlowUnitDesc>> descs) {
  flowunit_desc_ = std::move(descs);
}
}  // namespace modelbox
