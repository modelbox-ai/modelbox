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

#ifndef MODELBOX_FLOWUNIT_OUTPUT_BROKER_OBS_CPU_H_
#define MODELBOX_FLOWUNIT_OUTPUT_BROKER_OBS_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>

#include "eSDKOBS.h"
#include "obs_client.h"
#include "output_broker_flowunit.h"
#include "output_broker_plugin.h"


constexpr const char *DRIVER_NAME = "obs";
constexpr const char *DRIVER_DESC = "A obs output broker plugin on CPU";
constexpr const char *DRIVER_TYPE = "cpu";

constexpr const char *DATA_NAME = "data_name";

typedef struct tag_OBSOutputInfo {
  std::string ak;  // temporary USER AK, would be destoyed after use
  std::string sk;  // temporary USER SK, would be destoyed after use
  std::string
      token;  // temporary USER Security Token, would be destoyed after use
  std::string end_point;    // OBS EndPoint, for example:
                            // obs.cn-north-7.ulanqab.huawei.com
  std::string bucket;       // Bucket where the target file locates, for ex
  std::string path;         // path to save data, for example: obs-test/data/
  std::string domain_name;  // domain name of the resources agent
  std::string xrole_name;   // commit name
  std::string user_id;
  unsigned int file_key_index;  //
} OBSOutputInfo;

using ObsOutputConfigurations =
    std::map<std::string, std::shared_ptr<OBSOutputInfo>>;

class ObsOutputBroker : public OutputBrokerPlugin {
 public:
  ObsOutputBroker();
  ~ObsOutputBroker() override;

  modelbox::Status Init(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Deinit() override;

  /**
   * @brief Initial each broker: 1. allocate a broker id; 2. save the
   * configuration to a map
   * @param config - configurations in json style
   * @return a handle
   */
  std::shared_ptr<OutputBrokerHandle> Open(const std::string &config) override;

  /**
   * @brief Write data to target output
   * @param params - some parameters needed, as example:
   *                 (optional) "data_name": indicates the file key for the
   * uploaded data.
   * @return Successful or not
   */
  modelbox::Status Write(
      const std::shared_ptr<OutputBrokerHandle> &handle,
      const std::shared_ptr<modelbox::Buffer> &buffer) override;

  modelbox::Status Sync(
      const std::shared_ptr<OutputBrokerHandle> &handle) override;

  /**
   * @brief Remove the configuration
   * @param handle - to identify the configuration
   * @return Successful or not
   */
  modelbox::Status Close(
      const std::shared_ptr<OutputBrokerHandle> &handle) override;

 private:
  /**
   * @brief Print obs configuration
   * @param opt - obs configuration
   */
  void PrintObsConfig(const modelbox::ObsOptions &opt);

  ObsOutputConfigurations output_configs_;
  std::mutex output_cfgs_mutex_;
};

class OBSOutputBrokerFactory : public modelbox::DriverFactory {
 public:
  OBSOutputBrokerFactory() = default;
  ~OBSOutputBrokerFactory() override = default;

  std::shared_ptr<modelbox::Driver> GetDriver() override {
    std::shared_ptr<modelbox::Driver> parser =
        std::make_shared<ObsOutputBroker>();
    return parser;
  }
};

#endif  // MODELBOX_FLOWUNIT_OUTPUT_BROKER_OBS_CPU_H_
