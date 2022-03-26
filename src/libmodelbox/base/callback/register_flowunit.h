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
  ~RegisterFlowUnit();

  Status Open(const std::shared_ptr<Configuration> &config);

  /* class when unit is close */
  Status Close();

  Status Process(std::shared_ptr<DataContext> data_context);

  void SetCallBack(
      std::function<StatusCode(std::shared_ptr<DataContext>)> callback);

  std::function<StatusCode(std::shared_ptr<DataContext>)> GetCallBack();

 private:
  std::string name_;
  std::function<StatusCode(std::shared_ptr<DataContext>)> callback_{nullptr};
};

class RegisterFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  RegisterFlowUnitFactory() = default;
  RegisterFlowUnitFactory(
      const std::string unit_name, const std::set<std::string> inputs,
      const std::set<std::string> outputs,
      std::function<StatusCode(std::shared_ptr<DataContext>)> &callback);
  virtual ~RegisterFlowUnitFactory() = default;

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>
  FlowUnitProbe() {
    return desc_map_;
  }

  const std::string GetFlowUnitFactoryType() { return FLOWUNIT_TYPE; };
  std::shared_ptr<modelbox::FlowUnit> CreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type) override;

 private:
  Status Init();
  std::string unit_name_;
  std::set<std::string> input_ports_;
  std::set<std::string> output_ports_;
  std::function<StatusCode(std::shared_ptr<DataContext>)> callback_;
  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>> desc_map_;
};

}  // namespace modelbox
#endif
