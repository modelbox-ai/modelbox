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

#ifndef MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_OBS_CPU_H_
#define MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_OBS_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>

#include "data_source_parser_plugin.h"
#include "eSDKOBS.h"


#define OBS_TEMP_PATH "/tmp/ObsDownload/"

constexpr const char *DRIVER_NAME = "obs";
constexpr const char *DRIVER_DESC = "An OBS data source parser plugin on CPU";
constexpr const char *DRIVER_TYPE = "cpu";

typedef struct tag_OBSDownloadInfo {
  std::string ak;           // temporary USER AK
  std::string sk;           // temporary USER SK
  std::string token;        // temporary USER Security Token
  std::string domain_name;  // user/isv's domain name
  std::string xrole_name;   // agency name to vas
  std::string user_id;
  std::string end_point;  // OBS EndPoint, for example:
                          // obs.cn-north-7.ulanqab.huawei.com
  std::string bucket;     // Bucket where the target file locates, for ex
  std::string file_key;   // File Key, for example: obs-test/data/video.flv
  std::string file_local_path;  // local path of the downloaded file
} OBSDownloadInfo;

class ObsSourceParser : public DataSourceParserPlugin {
 public:
  ObsSourceParser();
  virtual ~ObsSourceParser();

  modelbox::Status Init(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Deinit() override;

  modelbox::Status Parse(const std::string &config, std::string &uri,
                         DestroyUriFunc &destroy_uri_func) override;
  modelbox::Status GetStreamType(const std::string &config,
                                 std::string &stream_type) override;

 private:
  /**
   * @brief Get EndPoint/Bucket/FileKey in the config.
   * @param download_info infos needed to download an object.
   * @param config configuration string
   * @return Successful or not
   */
  modelbox::Status GetObsInfo(OBSDownloadInfo &download_info,
                              const std::string &config);

  /**
   * @brief Generate a time-string, yyyymmddhhmmss
   * @param time in - the time to be converted to a string
   * @return Successful or not
   */
  std::string GetTimeString(time_t *time);

  std::string read_type_;
};

class ObsSourceParserFactory : public modelbox::DriverFactory {
 public:
  ObsSourceParserFactory() = default;
  virtual ~ObsSourceParserFactory() = default;

  std::shared_ptr<modelbox::Driver> GetDriver() override {
    std::shared_ptr<modelbox::Driver> parser =
        std::make_shared<ObsSourceParser>();
    return parser;
  }
};

#endif  // MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_OBS_CPU_H_
