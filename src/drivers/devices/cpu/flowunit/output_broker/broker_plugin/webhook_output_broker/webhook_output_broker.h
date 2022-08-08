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

 
#ifndef MODELBOX_FLOWUNIT_WEBHOOK_OUTPUT_BROKER_CPU_H_
#define MODELBOX_FLOWUNIT_WEBHOOK_OUTPUT_BROKER_CPU_H_
 
#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
 
#include "cpprest/http_client.h"
#include <modelbox/output_broker_plugin.h>
 
constexpr const char *DRIVER_NAME = "webhook";
constexpr const char *DRIVER_DESC = "A webhook output broker plugin on CPU";
constexpr const char *DRIVER_TYPE = "cpu";

typedef struct tag_WebhookOutputInfo {
  std::string url;
  std::map<std::string, std::string> headers;
}WebhookOutputInfo;

class WebhookOutputBroker : public modelbox::OutputBrokerPlugin {
 public:
  WebhookOutputBroker();

  ~WebhookOutputBroker() override;

  modelbox::Status Init(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
 
  modelbox::Status Deinit() override;
 
  std::shared_ptr<modelbox::OutputBrokerHandle> Open(const std::string &config) override;
 
  modelbox::Status Write(const std::shared_ptr<modelbox::OutputBrokerHandle> &handle,
                       const std::shared_ptr<modelbox::Buffer> &buffer) override;
 
  modelbox::Status Sync(
      const std::shared_ptr<modelbox::OutputBrokerHandle> &handle) override;
 
  modelbox::Status Close(
      const std::shared_ptr<modelbox::OutputBrokerHandle> &handle) override;
 
 private:
  modelbox::Status ParseConfig(const std::shared_ptr<modelbox::OutputBrokerHandle> &handle,
                             const std::string &config);
  std::map<std::string, std::shared_ptr<WebhookOutputInfo>> output_configs_;
  std::mutex output_configs_lock_;
  std::shared_ptr<web::http::client::http_client> client_;
};
 
class WebhookOutputBrokerFactory : public modelbox::DriverFactory {
 public:
  WebhookOutputBrokerFactory() = default;

  ~WebhookOutputBrokerFactory() override = default;

  std::shared_ptr<modelbox::Driver> GetDriver() override {
    std::shared_ptr<modelbox::Driver> parser =
        std::make_shared<WebhookOutputBroker>();
    return parser;
  }
};
 
#endif  // MODELBOX_FLOWUNIT_WEBHOOK_OUTPUT_BROKER_CPU_H_