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

#ifndef MODELBOX_FLOWUNIT_DIS_OUTPUT_BROKER_CPU_H_
#define MODELBOX_FLOWUNIT_DIS_OUTPUT_BROKER_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>

#include "dis/dis.h"
#include "output_broker_flowunit.h"
#include <modelbox/output_broker_plugin.h>

constexpr const char *DRIVER_NAME = "dis";
constexpr const char *DRIVER_DESC = "A dis output broker plugin on CPU";
constexpr const char *DRIVER_TYPE = "cpu";

typedef struct tag_DisOutputInfo {
  std::string ak;
  std::string sk;
  std::string token;
  std::string end_point;
  std::string region;
  std::string stream_name;
  std::string project_id;
  std::string domain_name;
  std::string xrole_name;
  std::string user_id;
} DisOutputInfo;

using DisOutputConfigurations =
    std::map<std::string, std::shared_ptr<DisOutputInfo>>;

class DisOutputBroker : public modelbox::OutputBrokerPlugin {
 public:
  DisOutputBroker();
  ~DisOutputBroker() override;

  modelbox::Status Init(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Deinit() override;

  std::shared_ptr<modelbox::OutputBrokerHandle> Open(const std::string &config) override;

  modelbox::Status Write(
      const std::shared_ptr<modelbox::OutputBrokerHandle> &handle,
      const std::shared_ptr<modelbox::Buffer> &buffer) override;

  modelbox::Status Sync(
      const std::shared_ptr<modelbox::OutputBrokerHandle> &handle) override;

  modelbox::Status Close(
      const std::shared_ptr<modelbox::OutputBrokerHandle> &handle) override;

 private:
  static DISStatus GetUserAuthInfo(char *project_id, char *ak_array,
                                   char *sk_array, char *x_security_token);
  static modelbox::Status GetCertInfo(
      std::shared_ptr<DisOutputInfo> &output_info);
  modelbox::Status ParseConfig(
      const std::shared_ptr<modelbox::OutputBrokerHandle> &handle,
      const std::string &config);
  static DISStatus PutRecordCallBack(char *error_code, char *error_details,
                                     char *stream_name,
                                     DISPutRecord *put_record, char *seq_number,
                                     char *partitiod_id);
  modelbox::Status JudgeTryAgain(long http_response_code);
  bool JudgeUpdateCert(long http_response_code);
  static DisOutputConfigurations output_configs_;
  static std::mutex output_configs_lock_;
  static std::atomic_bool init_flag_;
};

class DisOutputBrokerFactory : public modelbox::DriverFactory {
 public:
  DisOutputBrokerFactory() = default;
  virtual ~DisOutputBrokerFactory() = default;

  std::shared_ptr<modelbox::Driver> GetDriver() override {
    std::shared_ptr<modelbox::Driver> parser =
        std::make_shared<DisOutputBroker>();
    return parser;
  }
};

#endif  // MODELBOX_FLOWUNIT_DIS_OUTPUT_BROKER_CPU_H_