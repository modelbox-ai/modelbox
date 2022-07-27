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

#include "src/modelbox/serving/serving.h"

#include <dlfcn.h>
#include <stdio.h>
#include "modelbox/base/popen.h"

#include <fstream>

#include "gtest/gtest.h"
#include "mockflow.h"
#include "test_config.h"

namespace modelbox {

static std::set<std::string> SUPPORT_TF_VERSION = {"1.13.1", "1.15.0",
                                                   "2.6.0-dev20210809"};

class ModelboxServingTest : public testing::Test {
 public:
 protected:
  ModelboxServingTest() {
    flow_ = std::make_shared<Flow>();
  };

  virtual ~ModelboxServingTest() {};

  virtual void SetUp() {
    auto version = GetTFVersion();

    if (SUPPORT_TF_VERSION.find(version) == SUPPORT_TF_VERSION.end()) {
      version_suitable_ = false;
      MBLOG_INFO << "the version is " << version
                 << ", not in support version, skip test suit";
      GTEST_SKIP();
    }

    UpdateToml(version);
  };

  virtual void TearDown() {
    if (!version_suitable_) {
      GTEST_SKIP();
    }
  };

  const std::string test_lib_dir = TEST_DRIVER_DIR,
                    test_data_dir = TEST_DATA_DIR, test_assets = TEST_ASSETS,
                    test_toml_file = "test_serving_model.toml",
                    test_python_file = "test_custom_service.py";

  std::string default_custom_service_path, custom_service_path,
      new_test_serving_toml;

  void PrepareFiles(const std::string &default_serving_path, bool is_default);
  void RemoveFiles(const std::string &default_serving_path, const std::string &model_name);
  void ReplaceGraphToml(const std::string &graph_toml, const std::string &model_name,
                  std::string &update_graph_toml_file);
  modelbox::Status BuildAndRunFlow(const std::string &graph_path, int timeout = 10 * 1000);
 private:
  void UpdateToml(const std::string &version);
  modelbox::Status ReplaceVersion(const std::string &src,
                                  const std::string &dest,
                                  const std::string &version);
  std::string GetTFVersion();
  bool version_suitable_{true};
  std::shared_ptr<Flow> flow_;
};

void ModelboxServingTest::RemoveFiles(const std::string &default_serving_path, const std::string &model_name) {
    std::string default_graph_toml = "/tmp/" + model_name + "_new.toml";
    std::string default_flow_path = "/tmp/" + model_name;
    modelbox::RemoveDirectory(default_serving_path);
    modelbox::RemoveDirectory(default_flow_path);
    remove(default_graph_toml.c_str());
}


modelbox::Status ModelboxServingTest::ReplaceVersion(
    const std::string &src, const std::string &dest,
    const std::string &version) {
  if (access(dest.c_str(), F_OK) == 0) {
    if (remove(dest.c_str()) == -1) {
        return modelbox::STATUS_FAULT;
    }
  }

  std::ifstream src_file(src, std::ios::binary);
  std::ofstream dst_file(dest, std::ios::binary | std::ios::trunc);

  if (src_file.fail() || dst_file.fail()) {
    return modelbox::STATUS_FAULT;
  }

  std::string line;
  std::string tf_version = "TF_VERSION";

  while (std::getline(src_file, line)) {
    auto pos = line.find(tf_version);
    if (pos != std::string::npos) {
      line.replace(pos, tf_version.size(), version);
    }
    dst_file << line << "\n";
  }

  src_file.close();
  if (dst_file.fail()) {
    dst_file.close();
    remove(dest.c_str());
    return modelbox::STATUS_FAULT;
  }
  dst_file.close();

  return modelbox::STATUS_OK;
}

std::string ModelboxServingTest::GetTFVersion() {
  std::string ans = "";
  void *handler =
      dlopen(MODELBOX_TF_SO_PATH, RTLD_LOCAL | RTLD_DEEPBIND);
  if (handler == nullptr) {
    MBLOG_ERROR << "dlopen error: " << dlerror();
    return ans;
  }

  Defer { dlclose(handler); };
  typedef const char *(*TF_Version)();
  TF_Version func = nullptr;

  func = (TF_Version)dlsym(handler, "TF_Version");
  if (func == nullptr) {
    MBLOG_ERROR << "dlsym TF_Version failed, " << dlerror();
    return ans;
  }

  ans = func();
  return ans;
}

void ModelboxServingTest::UpdateToml(const std::string &version) {
  const std::string src_test_serving_toml =
      test_data_dir + "/" + test_toml_file;
  new_test_serving_toml = test_data_dir + "/test_serving_model_new.toml";
  auto status =
      ReplaceVersion(src_test_serving_toml, new_test_serving_toml, version);
  EXPECT_EQ(status, STATUS_OK);
}

void ModelboxServingTest::PrepareFiles(const std::string &default_serving_path,
                                       bool is_default) {
  auto mkdir_ret = mkdir(default_serving_path.c_str(), 0700);
  EXPECT_EQ(mkdir_ret, 0);

  const std::string dest_test_serving_toml =
      default_serving_path + "/model.toml";
  auto status = modelbox::CopyFile(new_test_serving_toml, dest_test_serving_toml);
  EXPECT_EQ(status, STATUS_OK);

  if (!is_default) {
    const std::string src_custom_service_file =
        test_data_dir + "/test_custom_service.py";
    const std::string dest_custom_sevice_file =
        default_serving_path + "/custom_service.py";
    auto status =
        modelbox::CopyFile(src_custom_service_file, dest_custom_sevice_file);
    EXPECT_EQ(status, STATUS_OK);
  }
}

void ModelboxServingTest::ReplaceGraphToml(const std::string &graph_toml,
                const std::string &model_name, std::string &update_graph_toml_file) {
  std::ifstream graph_reader(graph_toml);
  EXPECT_EQ(graph_reader.is_open(), true);
  
  update_graph_toml_file = "/tmp/" + model_name + "_new.toml";
  std::ofstream new_graph_writer(update_graph_toml_file);
  EXPECT_EQ(new_graph_writer.is_open(), true);
  
  std::stringstream ss;
  std::string content;
  while (std::getline(graph_reader, content)) {
    size_t pos;
    pos = content.find("skip-default");
    if (pos != std::string::npos) {
       ss << "skip-default = true\n";
       continue;
    }

   pos = content.find("dir");
   if (pos != std::string::npos) {
       ss << "dir=[\"" + std::string(TEST_DRIVER_DIR) + "\", \"/tmp/" + model_name + "\"]\n";
       continue;
   }

   ss << content << "\n";
  }

  new_graph_writer << ss.str();
  EXPECT_EQ(new_graph_writer.good(), true);
  graph_reader.close();
  new_graph_writer.close();
  EXPECT_EQ(remove(graph_toml.c_str()), 0);
}

modelbox::Status ModelboxServingTest::BuildAndRunFlow(const std::string &graph_path, int timeout) {
  auto ret = flow_->Init(graph_path);
  if (!ret) {
     return ret;
  }

  ret = flow_->Build();
  if (!ret) {
     return ret;
  }

  ret = flow_->RunAsync();
  if (!ret) {
     return ret;
  }

  if (timeout < 0) {
     return ret;
  }

  Status retval;
  flow_->Wait(timeout, &retval);
  return retval;
}

TEST_F(ModelboxServingTest, DefaultCustomService) {
  const std::string default_serving_path =
      test_data_dir + "/default_test_serving_path";
  std::string model_name = "test_default_custom_service";
  PrepareFiles(default_serving_path, true);
  auto serving = std::make_shared<ModelServing>();
  auto status = serving->GenerateTemplate(model_name,
                                          default_serving_path, 39110);
  EXPECT_EQ(status, STATUS_OK);
  std::string default_graph_toml = "/tmp/" + model_name + ".toml";
  std::string default_graph_toml_new;
  ReplaceGraphToml(default_graph_toml, model_name, default_graph_toml_new);
  status = BuildAndRunFlow(default_graph_toml_new);
  EXPECT_EQ(status, STATUS_OK);
  modelbox::Popen p;
  std::vector<std::string> cmds{"python3"};
  std::string python_file = test_data_dir + "/test_client.py";
  cmds.push_back(python_file);
  p.Open(cmds);
  std::string line;
  p.ReadOutLine(line);
  auto res_line = line.substr(0, line.size() - 1);  
  EXPECT_EQ(res_line, "{\"output\": [1.05097496509552, 1.3005822896957397, 1.550189733505249]}");
  auto ret = p.Close();
  EXPECT_EQ(WEXITSTATUS(ret), 0);
  RemoveFiles(default_serving_path, model_name);
}

TEST_F(ModelboxServingTest, CustomService) {
  const std::string serving_path = test_data_dir + "/test_serving_path";
  std::string model_name = "test_custom_service";
  PrepareFiles(serving_path, false);
  auto serving = std::make_shared<ModelServing>();
  auto status =
      serving->GenerateTemplate(model_name, serving_path, 39110);
  EXPECT_EQ(status, STATUS_OK);
  std::string default_graph_toml = "/tmp/" + model_name + ".toml";
  std::string default_graph_toml_new;
  ReplaceGraphToml(default_graph_toml, model_name, default_graph_toml_new);
  status = BuildAndRunFlow(default_graph_toml_new);
  EXPECT_EQ(status, STATUS_OK);
  modelbox::Popen p;
  std::vector<std::string> cmds{"python3"};
  std::string python_file = test_data_dir + "/test_client.py";
  cmds.push_back(python_file);
  p.Open(cmds);
  std::string line;
  p.ReadOutLine(line);
  auto res_line = line.substr(0, line.size() - 1);
  EXPECT_EQ(res_line, "{\"predict_result\": \"2\"}");
  auto ret = p.Close();
  EXPECT_EQ(WEXITSTATUS(ret), 0);
  RemoveFiles(serving_path, model_name);
}

}  // namespace modelbox
