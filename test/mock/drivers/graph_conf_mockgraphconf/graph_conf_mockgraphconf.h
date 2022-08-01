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


#ifndef MODELBOX_GRAPHMANAGER_MOCK_GRAPHCONF_H_
#define MODELBOX_GRAPHMANAGER_MOCK_GRAPHCONF_H_

#include <modelbox/base/graph_manager.h>
#include <modelbox/flow.h>

#include "gmock/gmock.h"
#include "graphviz_conf.h"
#include "gtest/gtest.h"
#include "mock_driver_ctl.h"

namespace modelbox {

constexpr const char *MOCK_GRAPHCONF_TYPE = "MOCKGRAPHVIZ";
constexpr const char *MOCK_GRAPHCONF_NAME = "MOCK-GRAPHCONF-GRAPHVIZ";
constexpr const char *MOCK_GRAPHCONF_DESC = "mock graph config parse graphviz";

class MockGraphConfig : public modelbox::GraphConfig {
 public:
  using GraphConfig::Resolve;
  MockGraphConfig(const std::string &graph_conf_path) {
    EXPECT_CALL(*this, Resolve)
        .WillRepeatedly([this](std::shared_ptr<modelbox::GCGraph> graph) {
          return this->Resolve(graph);
        });
  };
  ~MockGraphConfig() override = default;

  MOCK_METHOD(bool, Resolve, (std::shared_ptr<modelbox::GCGraph>));

  std::shared_ptr<modelbox::GraphConfig> graphconfig_;
};

class MockGraphConfigFactory : public modelbox::GraphConfigFactory {
 public:
  MockGraphConfigFactory() {
    EXPECT_CALL(*this, CreateGraphConfigFromStr)
        .WillRepeatedly([this](const std::string &config_path) {
          bind_factory_ = std::make_shared<modelbox::GraphvizFactory>();
          return bind_factory_->CreateGraphConfigFromStr(config_path);
        });

    EXPECT_CALL(*this, CreateGraphConfigFromFile)
        .WillRepeatedly([this](const std::string &file_path) {
          bind_factory_ = std::make_shared<modelbox::GraphvizFactory>();
          return bind_factory_->CreateGraphConfigFromFile(file_path);
        });

    EXPECT_CALL(*this, GetGraphConfFactoryType).WillRepeatedly([this]() {
      bind_factory_ = std::make_shared<modelbox::GraphvizFactory>();
      return bind_factory_->GetGraphConfFactoryType();
    });
  };

  ~MockGraphConfigFactory() override = default;

  MOCK_METHOD(std::shared_ptr<GraphConfig>, CreateGraphConfigFromStr,
              (const std::string &config_path));
  MOCK_METHOD(std::shared_ptr<GraphConfig>, CreateGraphConfigFromFile,
              (const std::string &file_path));
  MOCK_METHOD(std::string, GetGraphConfFactoryType, ());

 private:
  std::shared_ptr<GraphvizFactory> bind_factory_;
};

class MockDriverGraphConfig : public modelbox::MockDriver {
 public:
  MockDriverGraphConfig() = default;
  ~MockDriverGraphConfig() override = default;

  static MockDriverGraphConfig *Instance() { return &desc_; };

 private:
  static MockDriverGraphConfig desc_;
};

}  // namespace modelbox

#endif  // MODELBOX_GRAPHMANAGER_MOCK_GRAPHCONF_H_