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

#ifndef CALLBACK_FLOWUNIT_H_
#define CALLBACK_FLOWUNIT_H_
#include <modelbox/flowunit.h>

#define FLOWUNIT_TYPE "cpu"

namespace modelbox {

class RegisterFlowUnit : public FlowUnit {
 public:
  RegisterFlowUnit(const std::string &name);
  ~RegisterFlowUnit() override;

  Status Open(const std::shared_ptr<Configuration> &config) override;

  /* class when unit is close */
  Status Close() override;

  Status Process(std::shared_ptr<DataContext> data_context) override;

  void SetCallBack(
      std::function<Status(std::shared_ptr<DataContext>)> callback);

  std::function<Status(std::shared_ptr<DataContext>)> GetCallBack();

 private:
  std::string name_;
  std::function<Status(std::shared_ptr<DataContext>)> callback_{nullptr};
};

class RegisterFlowUnitFactory : public FlowUnitFactory {
 public:
  RegisterFlowUnitFactory();
  RegisterFlowUnitFactory(
      std::string unit_name, std::vector<std::string> inputs,
      std::vector<std::string> outputs,
      std::function<Status(std::shared_ptr<DataContext>)> callback);
  ~RegisterFlowUnitFactory() override;

  std::map<std::string, std::shared_ptr<FlowUnitDesc>> FlowUnitProbe() override;

  std::string GetFlowUnitFactoryType() override;

  std::string GetFlowUnitFactoryName() override;

  std::shared_ptr<FlowUnit> CreateFlowUnit(
      const std::string &name, const std::string &unit_type) override;

 private:
  Status Init();
  std::string unit_name_;
  std::vector<std::string> input_ports_;
  std::vector<std::string> output_ports_;
  std::function<Status(std::shared_ptr<DataContext>)> callback_;
  std::map<std::string, std::shared_ptr<FlowUnitDesc>> desc_map_;
};

}  // namespace modelbox
#endif
