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

#ifndef MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_CPU_H_
#define MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "data_source_parser_plugin.h"
#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

constexpr const char *FLOWUNIT_NAME = "data_source_parser";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: this flowunit can obtain the video stream address or download "
    "the video file to the local according to the input configuration data, "
    "and output the url. Currently supported types have obs, vcn, vis, "
    "resetful, url. \n"
    "\t@Port parameter: The input buffer data type is char *, and contain the "
    "following meta fields:\n"
    "\t\tField Name: source_type,   Type: string\n"
    "\t  the output buffer data type is char *. \n"
    "\t@Constraint: the field value range of this flowunit "
    "support: 'source_type': "
    "[obs, vcn, vis, restful, url]. This flowunit is usually followed by "
    "'video_demuxer'.";
constexpr const char *INPUT_DATA_SOURCE_CFG = "in_data";
constexpr const char *INPUT_META_SOURCE_TYPE = "source_type";
constexpr const char *OUTPUT_STREAM_META = "out_video_url";
constexpr const char *STREAM_META_SOURCE_URL = "source_url";
constexpr const char *SOURCE_PARSER_FLOWUNIT = "source_parser_flowunit";
constexpr const char *PARSER_RETRY_CONTEXT = "source_context";

class DataSourceParserFlowUnit : public modelbox::FlowUnit {
 public:
  DataSourceParserFlowUnit();
  ~DataSourceParserFlowUnit() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;

  modelbox::Status Close() override;

  /* run when processing data */
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  std::shared_ptr<modelbox::SourceContext> Parse(
      std::shared_ptr<modelbox::SessionContext> session_context,
      const std::string &source_type, const std::string &data_source_cfg,
      std::shared_ptr<std::string> &uri);

  std::shared_ptr<DataSourceParserPlugin> GetPlugin(
      const std::string &source_type);

  modelbox::Status WriteData(
      std::shared_ptr<modelbox::DataContext> &data_ctx,
      const std::shared_ptr<std::string> &uri, const std::string &source_type,
      const std::string &data_source_cfg,
      std::shared_ptr<modelbox::SourceContext> &source_context);

  std::vector<std::shared_ptr<modelbox::DriverFactory>> factories_;
  std::map<std::string, std::shared_ptr<DataSourceParserPlugin>> plugins_;
};

#endif  // MODELBOX_FLOWUNIT_DATA_SOURCE_PARSER_CPU_H_
