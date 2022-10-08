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

#include "modelbox/common/config.h"

#include <linux/limits.h>
#include <modelbox/base/configuration.h>
#include <unistd.h>

namespace modelbox {

void PluginMergeUniq(std::vector<std::string> *to,
                     std::vector<std::string> *from) {
  if (from->size() == 0) {
    return;
  }

  for (const auto &t : *to) {
    for (auto itr = from->begin(); itr != from->end(); itr++) {
      if (t == *itr) {
        from->erase(itr);
        break;
      }
    }
  }

  to->insert(to->end(), from->begin(), from->end());
}

std::shared_ptr<modelbox::Configuration> LoadSubConfig(
    const std::string &file) {
  modelbox::ConfigurationBuilder config_builder;

  auto curr_config = config_builder.Build(file, ConfigType::TOML, true);
  if (curr_config == nullptr) {
    MBLOG_ERROR << "Load config file " << file
                << " failed, detail: " << modelbox::StatusError.Errormsg();
    fprintf(stderr, "Load config %s failed, detail:\n", file.c_str());
    fprintf(stderr, "%s\n", modelbox::StatusError.Errormsg().c_str());
    return nullptr;
  }

  auto include_conf_files = curr_config->GetStrings("include.files");
  curr_config->SetProperty("include.files", std::vector<std::string>());

  auto cur_conf_dir = modelbox::GetDirName(file);
  if (cur_conf_dir.length() <= 0 || cur_conf_dir == ".") {
    char cwd[PATH_MAX];
    cur_conf_dir = getcwd(cwd, sizeof(cwd));
  }

  for (const auto &conf_file_pattern : include_conf_files) {
    auto pattern_conf_dir = modelbox::GetDirName(conf_file_pattern);
    auto pattern_file = modelbox::GetBaseName(conf_file_pattern);
    std::vector<std::string> conf_files;
    std::string include_conf_dir;
    if (pattern_conf_dir.length() > 0 && pattern_conf_dir != ".") {
      include_conf_dir = pattern_conf_dir;
    } else {
      include_conf_dir = cur_conf_dir;
    }

    modelbox::ListFiles(include_conf_dir, pattern_file, &conf_files,
                        modelbox::LIST_FILES_FILE);
    std::sort(conf_files.begin(), conf_files.end());
    for (const auto &conf_file : conf_files) {
      auto conf = LoadSubConfig(conf_file);
      if (conf == nullptr) {
        continue;
      }

      auto plugins = curr_config->GetStrings("plugin.files");
      auto include_plugins = conf->GetStrings("plugin.files");

      PluginMergeUniq(&plugins, &include_plugins);
      curr_config->Add(*conf);
      curr_config->SetProperty("plugin.files", plugins);
    }
  }

  return curr_config;
}

}  // namespace modelbox