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

#define _TURN_OFF_PLATFORM_STRING

#include "src/modelbox/server/server.h"

#include <dlfcn.h>
#include <ftw.h>
#include <modelbox/base/popen.h>
#include <stdio.h>

#include <fstream>
#include <future>
#include <nlohmann/json.hpp>

#include "gtest/gtest.h"
#include "memory"
#include "mock_server.h"
#include "mock_tool.h"
#include "mockflow.h"
#include "modelbox/server/job_manager.h"
#include "src/modelbox/server/config.h"
#include "test_config.h"
#include "thread"

extern char **environ;

namespace modelbox {

class ModelboxServerTest : public testing::Test {
 public:
  ModelboxServerTest() {}

 protected:
  virtual void SetUp() {
    flow_ = std::make_shared<MockFlow>();
    flow_->Init(false);
    flow_->Register_Test_0_2_Flowunit();
    flow_->Register_Test_OK_2_0_Flowunit();
  };
  virtual void TearDown() { flow_->Destroy(); };

 private:
  std::shared_ptr<MockFlow> flow_;
};

TEST_F(ModelboxServerTest, Post) {
  std::shared_ptr<JobManager> job_manager = std::make_shared<JobManager>();
  auto job = job_manager->CreateJob("test", "");

  EXPECT_EQ(job->GetJobName(), "test");
}

int rmfiles(const char *pathname, const struct stat *sbuf, int type,
            struct FTW *ftwb) {
  remove(pathname);
  return 0;
}

void removedir(const std::string &path) {
  nftw(path.c_str(), rmfiles, 10, FTW_DEPTH | FTW_MOUNT | FTW_PHYS);
}

nlohmann::json GetCreateJobMsg(const std::string &name) {
  const std::string test_lib_dir = TEST_LIB_DIR;
  auto graph = R"(
      digraph demo {
      IN[flowunit=test_0_2]
      OUT[flowunit=test_ok_2_0]
      IN:Out_1->OUT:In_1
      IN:Out_2->OUT:In_2
  })";

  auto create_body = nlohmann::json::parse(R"(
      {
        "job_id" : "",
        "job_graph_format" : "json",
        "job_graph": {
          "driver": {
            "skip-default": true
          },
          "graph": {
            "graphconf" : "",
            "format":"graphviz"
          }
        }
      }
    )");
  create_body["job_id"] = name;
  create_body["job_graph"]["driver"]["dir"] = test_lib_dir;
  create_body["job_graph"]["graph"]["graphconf"] = graph;
  return create_body;
}

nlohmann::json GetCreateJobFail(const std::string &name) {
  const std::string test_lib_dir = TEST_LIB_DIR;
  auto graph = R"(
      digraph demo {
      IN[flowunit=not_exist_in]
      OUT[flowunit=not_exist_out]
      IN:Out_1->OUT:In_1
      IN:Out_2->OUT:In_2
  })";

  auto create_body = nlohmann::json::parse(R"(
      {
        "job_id" : "",
        "job_graph_format" : "json",
        "job_graph": {
          "driver": {
            "skip-default": true
          },
          "graph": {
            "graphconf" : "",
            "format":"graphviz"
          }
        }
      }
    )");
  create_body["job_id"] = name;
  create_body["job_graph"]["driver"]["dir"] = test_lib_dir;
  create_body["job_graph"]["graph"]["graphconf"] = graph;
  return create_body;
}

nlohmann::json GetFlowInfoMsg(const std::vector<std::string> &dir_list) {
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  auto body = nlohmann::json::parse(R"(
      {
        "skip-default" : true,
        "dir" : []
      }
    )");

  auto dirs = nlohmann::json::array();
  for (const auto &dir : dir_list) {
    dirs.push_back(dir);
  }
  body["dir"] = dirs;
  return body;
}

httplib::Response CreateJob(MockServer &server, const nlohmann::json &body) {
  HttpRequest request(HttpMethods::PUT,
                      server.GetServerURL() + "/v1/modelbox/job/");
  request.SetBody(body.dump());

  return server.DoRequest(request);
}

httplib::Response ListAllJobs(MockServer &server) {
  HttpRequest request(HttpMethods::GET,
                      server.GetServerURL() + "/v1/modelbox/job/list/all");
  return server.DoRequest(request);
}

httplib::Response QueryJob(MockServer &server, const std::string &name) {
  HttpRequest request(HttpMethods::GET,
                      server.GetServerURL() + "/v1/modelbox/job/" + name);
  return server.DoRequest(request);
}

httplib::Response DeleteJob(MockServer &server, const std::string &name) {
  HttpRequest request(HttpMethods::DELETE,
                      server.GetServerURL() + "/v1/modelbox/job/" + name);
  return server.DoRequest(request);
}

httplib::Response GetFlowInfo(MockServer &server) {
  HttpRequest request(HttpMethods::GET,
                      server.GetServerURL() + "/editor/flow-info");
  return server.DoRequest(request);
}

httplib::Response GetFlowInfoSpecificDir(
    MockServer &server, const std::vector<std::string> &dir_list) {
  HttpRequest request(HttpMethods::PUT,
                      server.GetServerURL() + "/editor/flow-info");
  request.SetBody(GetFlowInfoMsg(dir_list).dump());
  return server.DoRequest(request);
}

httplib::Response EditorCreateProject(MockServer &server,
                                      const std::string &projectname,
                                      const std::string &path) {
  HttpRequest request(HttpMethods::PUT,
                      server.GetServerURL() + "/editor/project/create");
  nlohmann::json create_body;
  create_body["name"] = projectname;
  create_body["path"] = path;
  request.SetBody(create_body.dump());
  return server.DoRequest(request);
}

httplib::Response EditorQueryProject(MockServer &server,
                                     const std::string &path) {
  HttpRequest request(HttpMethods::GET,
                      server.GetServerURL() + "/editor/project?path=" + path);
  return server.DoRequest(request);
}

httplib::Response EditorCreateFlowunit(
    MockServer &server, const std::map<std::string, std::string> &value,
    const std::vector<std::string> &in = {},
    const std::vector<std::string> &out = {}) {
  HttpRequest request(HttpMethods::PUT,
                      server.GetServerURL() + "/editor/flowunit/create");
  nlohmann::json create_body;
  size_t i = 0;

  for (const auto &kv : value) {
    create_body[kv.first] = kv.second;
  }

  nlohmann::json injson;
  for (i = 0; i < in.size(); i++) {
    injson["name"] = in[i];
  }

  if (in.size() == 0) {
    create_body["input"] = "-";
  } else {
    create_body["input"] = injson;
  }

  nlohmann::json outjson;
  for (i = 0; i < out.size(); i++) {
    outjson["name"] = out[i];
  }

  if (out.size() == 0) {
    create_body["output"] = "-";
  } else {
    create_body["output"] = outjson;
  }
  request.SetBody(create_body.dump());
  return server.DoRequest(request);
}

httplib::Response GetDemo(MockServer &server, const std::string &demo = "") {
  auto url = server.GetServerURL() + "/editor/demo";
  if (!demo.empty()) {
    url = url + "/" + demo;
  }
  HttpRequest request(HttpMethods::GET, url);
  return server.DoRequest(request);
}

TEST_F(ModelboxServerTest, CreateJob) {
  MockServer server;
  auto ret = server.Init(nullptr);
  if (ret == STATUS_NOTSUPPORT) {
    GTEST_SKIP();
  }
  server.Start();
  sleep(1);
  auto body = GetCreateJobMsg("example");
  auto response = CreateJob(server, body);
  MBLOG_INFO << response.body;
  EXPECT_EQ(response.status, HttpStatusCodes::CREATED);
}

TEST_F(ModelboxServerTest, CreateJobFail) {
  MockServer server;
  auto ret = server.Init(nullptr);
  if (ret == STATUS_NOTSUPPORT) {
    GTEST_SKIP();
  }
  server.Start();
  sleep(1);
  auto body = GetCreateJobFail("example");
  auto response = CreateJob(server, body);
  MBLOG_INFO << response.body;
  EXPECT_EQ(response.status, HttpStatusCodes::BAD_REQUEST);
  EXPECT_NE(response.body.find_first_of("not_exist_in"), std::string::npos);
}

TEST_F(ModelboxServerTest, ListAllJobs) {
  MockServer server;
  auto ret = server.Init(nullptr);
  if (ret == STATUS_NOTSUPPORT) {
    GTEST_SKIP();
  }
  server.Start();
  sleep(1);
  auto body = GetCreateJobMsg("example1");
  auto create_response = CreateJob(server, body);
  EXPECT_EQ(create_response.status, HttpStatusCodes::CREATED);

  body = GetCreateJobMsg("example2");
  create_response = CreateJob(server, body);
  EXPECT_EQ(create_response.status, HttpStatusCodes::CREATED);

  auto response = ListAllJobs(server);
  EXPECT_EQ(response.status, HttpStatusCodes::OK);
  auto result = nlohmann::json::parse(response.body);
  auto jobs = result["job_list"];
  EXPECT_EQ(jobs[0]["job_id"], "example2");
  EXPECT_EQ(jobs[0]["job_status"], "RUNNING");
  EXPECT_EQ(jobs[1]["job_id"], "example1");
  EXPECT_EQ(jobs[1]["job_status"], "RUNNING");
}

TEST_F(ModelboxServerTest, QueryJobNotExists) {
  MockServer server;
  auto ret = server.Init(nullptr);
  if (ret == STATUS_NOTSUPPORT) {
    GTEST_SKIP();
  }
  server.Start();
  sleep(1);
  auto response = QueryJob(server, "example");
  EXPECT_EQ(response.status, HttpStatusCodes::NOT_FOUND);
}

TEST_F(ModelboxServerTest, QueryJob) {
  MockServer server;
  auto ret = server.Init(nullptr);
  if (ret == STATUS_NOTSUPPORT) {
    GTEST_SKIP();
  }
  server.Start();
  sleep(1);
  auto body = GetCreateJobMsg("example");
  auto create_response = CreateJob(server, body);
  EXPECT_EQ(create_response.status, HttpStatusCodes::CREATED);
  auto response = QueryJob(server, "example");
  EXPECT_EQ(response.status, HttpStatusCodes::OK);
  auto result = nlohmann::json::parse(response.body);
  EXPECT_EQ(result["job_id"], "example");
  EXPECT_EQ(result["job_status"], "RUNNING");
}

TEST_F(ModelboxServerTest, DeleteJob) {
  MockServer server;
  auto ret = server.Init(nullptr);
  if (ret == STATUS_NOTSUPPORT) {
    GTEST_SKIP();
  }
  server.Start();
  sleep(1);
  auto body = GetCreateJobMsg("example");
  auto response = CreateJob(server, body);
  EXPECT_EQ(response.status, HttpStatusCodes::CREATED);
  response = QueryJob(server, "example");
  EXPECT_EQ(response.status, HttpStatusCodes::OK);
  response = DeleteJob(server, "example");
  EXPECT_EQ(response.status, HttpStatusCodes::NO_CONTENT);
  response = QueryJob(server, "example");
  EXPECT_EQ(response.status, HttpStatusCodes::NOT_FOUND);
}

TEST_F(ModelboxServerTest, QueryDemo) {
  MockServer server;
  std::string demo_dir = std::string(TEST_DATA_DIR) + "/demo";
  CreateDirectory(demo_dir);
  Defer { remove(demo_dir.c_str()); };

  auto conf = std::make_shared<Configuration>();
  conf->SetProperty("editor.demo_graphs", demo_dir);

  auto create_file = [](const std::string &file, const std::string &content) {
    std::ofstream out(file, std::ios::trunc);
    if (out.fail()) {
      return false;
    }
    Defer { out.close(); };

    out << content;
    if (out.fail()) {
      return false;
    }
    return true;
  };

  create_file(demo_dir + "/flow1.json", "{\"key\":\"value\"}");
  create_file(demo_dir + "/flow2.toml", "key = \"value\"");

  auto ret = server.Init(conf);
  if (ret == STATUS_NOTSUPPORT) {
    GTEST_SKIP();
  }
  server.Start();
  sleep(1);
  auto response = GetDemo(server);
  EXPECT_EQ(response.status, HttpStatusCodes::OK);
  auto result = nlohmann::json::parse(response.body);
  MBLOG_INFO << response.body;
  auto demo_list = result["demo_list"];
  EXPECT_EQ(demo_list[0]["name"], "flow2.toml");
  EXPECT_EQ(demo_list[1]["name"], "flow1.json");

  response = GetDemo(server, "flow1.json");
  EXPECT_EQ(response.status, HttpStatusCodes::OK);
  result = nlohmann::json::parse(response.body);
  MBLOG_INFO << response.body;
  EXPECT_EQ(result["key"], "value");

  response = GetDemo(server, "flow2.toml");
  EXPECT_EQ(response.status, HttpStatusCodes::OK);
  MBLOG_INFO << response.body;
}

// Python library conflict problem, disable test cases.
TEST_F(ModelboxServerTest, DISABLED_FlowInfo) {
  MockServer server;
  auto ret = server.Init(nullptr);
  if (ret == STATUS_NOTSUPPORT) {
    GTEST_SKIP();
  }
  server.Start();
  sleep(1);
  auto response = GetFlowInfo(server);
  EXPECT_EQ(response.status, HttpStatusCodes::OK);
}

TEST_F(ModelboxServerTest, FlowInfoSpecificPath) {
  MockServer server;
  auto ret = server.Init(nullptr);
  if (ret == STATUS_NOTSUPPORT) {
    GTEST_SKIP();
  }
  server.Start();
  sleep(1);
  std::vector<std::string> dir_list;
  dir_list.push_back(VIRTUAL_PYTHON_PATH);
  auto response = GetFlowInfoSpecificDir(server, dir_list);
  EXPECT_EQ(response.status, HttpStatusCodes::OK);
}

TEST_F(ModelboxServerTest, TemplateCommandTest) {
  MockServer server;
  auto conf = std::make_shared<Configuration>();
  std::string template_env = "MODELBOX_TEMPLATE_PATH";
  std::string project_name = "test";
  template_env += "=" + std::string(MODELBOX_TEMPLATE_BIN_DIR);

  std::string tmp_path = TEST_WORKING_DIR + std::string("/tmp/project");
  removedir(tmp_path.c_str());
  Defer { removedir(tmp_path.c_str()); };

  conf->SetProperty("editor.test.template_cmd_env", template_env);
  conf->SetProperty("editor.test.template_cmd", MODELBOX_TEMPLATE_CMD_PATH);
  auto ret = server.Init(conf);
  if (ret == STATUS_NOTSUPPORT) {
    GTEST_SKIP();
  }
  server.Start();
  sleep(1);
  auto response = EditorCreateProject(server, project_name, tmp_path);
  MBLOG_INFO << response.body.c_str();
  EXPECT_EQ(response.status, HttpStatusCodes::CREATED);

  response = EditorCreateFlowunit(
      server, {{"name", "cpp"}, {"lang", "c++"}, {"project-path", tmp_path}});
  MBLOG_INFO << response.body.c_str();
  EXPECT_EQ(response.status, HttpStatusCodes::CREATED);

  response = EditorCreateFlowunit(
      server,
      {{"name", "python"}, {"lang", "python"}, {"project-path", tmp_path}});
  MBLOG_INFO << response.body.c_str();
  EXPECT_EQ(response.status, HttpStatusCodes::CREATED);

  response = EditorCreateFlowunit(server, {{"name", "infer"},
                                           {"lang", "infer"},
                                           {"project-path", tmp_path},
                                           {"virtual-type", "tensorrt"},
                                           {"model", "modelfile"}});
  MBLOG_INFO << response.body.c_str();
  EXPECT_EQ(response.status, HttpStatusCodes::CREATED);

  response =
      EditorCreateFlowunit(server, {{"name", "yolo"},
                                    {"lang", "yolo"},
                                    {"project-path", tmp_path},
                                    {"virtual-type", "yolov3_postprocess"}});
  MBLOG_INFO << response.body.c_str();
  EXPECT_EQ(response.status, HttpStatusCodes::CREATED);

  response = EditorQueryProject(server, tmp_path);
  MBLOG_INFO << response.body.c_str();
  EXPECT_EQ(response.status, HttpStatusCodes::OK);
  auto result = nlohmann::json::parse(response.body);
  EXPECT_EQ(result["project_name"], project_name);
  EXPECT_EQ(result["flowunits"].size(), 4);
}

TEST_F(ModelboxServerTest, ServerLoadConfig) {
  const std::string test_etc_dir = TEST_DATA_DIR;
  const std::string test_lib_dir = TEST_LIB_DIR;
  std::string conf = R"""([server]
ip = "127.0.0.1"
port = "1104"
plugin_path = ")""" + test_lib_dir +
                     R"""(/modelbox-plugin.so"
)""";

  MBLOG_INFO << "modelbox config: \n" << conf;
  std::string config_file_path = test_etc_dir + "/modelbox.conf";
  std::ofstream ofs(config_file_path);
  EXPECT_TRUE(ofs.is_open());
  ofs.write(conf.data(), conf.size());
  ofs.flush();
  ofs.close();
  Defer { remove(config_file_path.c_str()); };
  ASSERT_TRUE(LoadConfig(config_file_path));
}

TEST_F(ModelboxServerTest, Log) {
  MockServer server;
  MockTool tool;
  auto conf = std::make_shared<Configuration>();
  auto retval = server.Init(conf);
  if (retval == STATUS_NOTSUPPORT) {
    GTEST_SKIP();
  }
  server.Start();
  auto ret = tool.Run("server log --getlevel");
  EXPECT_EQ(ret, 0);
  ret = tool.Run("server log --setlevel debug");
  MBLOG_DEBUG << "You will see this log";
  EXPECT_EQ(ret, 0);
  ret = tool.Run("server log --getlevel");
  EXPECT_EQ(ret, 0);
  ret = tool.Run("server log --setlevel info");
  EXPECT_EQ(ret, 0);
  server.Stop();
}

TEST_F(ModelboxServerTest, Slab) {
  MockServer server;
  MockTool tool;
  auto conf = std::make_shared<Configuration>();
  auto retval = server.Init(conf);
  if (retval == STATUS_NOTSUPPORT) {
    GTEST_SKIP();
  }
  server.Start();

  auto ret = tool.Run("server slab");
  EXPECT_EQ(ret, 1);
  ret = tool.Run("server slab --device");
  EXPECT_EQ(ret, 0);
  ret = tool.Run("server slab --device --type cpu");
  EXPECT_EQ(ret, 0);
  ret = tool.Run("server slab --device --type cuda");
  EXPECT_EQ(ret, 1);
  ret = tool.Run("server slab --device --type cpu --id 0");
  EXPECT_EQ(ret, 0);
  ret = tool.Run("server slab --device --type cuda --id 0");
  EXPECT_EQ(ret, 1);
  sleep(1);
  server.Stop();
}

TEST_F(ModelboxServerTest, Stat) {
  MockServer server;
  MockTool tool;
  auto conf = std::make_shared<Configuration>();
  auto retval = server.Init(conf);
  if (retval == STATUS_NOTSUPPORT) {
    GTEST_SKIP();
  }
  server.Start();

  auto root = modelbox::Statistics::GetGlobalItem();
  // FlowUnit
  auto flow_item = root->AddItem(modelbox::STATISTICS_ITEM_FLOW);
  auto session_item = flow_item->AddItem("SessionId");
  auto decoder_item = session_item->AddItem("video_decoder");
  flow_item->SetValue<uint64_t>(1);
  // Device
  auto device_item = root->AddItem("Device");
  auto gpu0_item = device_item->AddItem("gpu0", 20);
  auto gpu1_item = device_item->AddItem("gpu1", 15);
  gpu1_item->SetValue<uint64_t>(10);

  auto ret = tool.Run("server stat --all");
  EXPECT_EQ(ret, 0);
  ret = tool.Run("server stat --node gpu1");
  EXPECT_EQ(ret, 0);
  ret = tool.Run("server stat --node flow");
  EXPECT_EQ(ret, 0);
  ret = tool.Run("server stat --node not_exist");
  EXPECT_EQ(ret, 0);
  server.Stop();
}

TEST_F(ModelboxServerTest, JSPlugin) {
  const std::string test_etc_dir = TEST_DATA_DIR;
  const std::string test_js_path = test_etc_dir + "/modelbox-plugin-billing.js";
  std::string conf = R"(
                     [server]
                     ip = "127.0.0.1"
                     port = "1104"
                     [plugin]
                     files = [")" +
                     test_js_path +
                     R"("]
                     )";

  MBLOG_INFO << "modelbox config: \n" << conf;
  std::string config_file_path = test_etc_dir + "/modelbox.conf";
  std::ofstream ofs(config_file_path);
  EXPECT_TRUE(ofs.is_open());
  ofs.write(conf.data(), conf.size());
  ofs.flush();
  ofs.close();
  Defer { remove(config_file_path.c_str()); };

  std::string js = R"(
    var bill_for_sessions = {};
    var value_count = {"value_count": 0};

    function init(ctx) {
      console.error("Register stats notify");
      var type = NOTIFY_CREATE | NOTIFY_CHANGE | NOTIFY_DELETE;
      registerStatsNotify("flow.*", type, "onGraphChange");
      registerStatsNotify("flow.*.*", type, "onSessionChange");
      registerStatsNotify("flow.*.*.demuxer.video_duration", type, "onValueChange", value_count);
      registerStatsNotify("flow.*.*.demuxer.video_rate", type, "onValueChange");
      registerStatsNotify("flow.*.*.decoder.width", type, "onValueChange");
      registerStatsNotify("flow.*.*.decoder.height", type, "onValueChange");
      registerStatsNotify("flow.*.*.decoder.frame_count", type, "onValueChange");
    }

    function onGraphChange(path, value, type) {
      if (type & NOTIFY_DELETE) {
        routeData("test_router", "bill", JSON.stringify(bill_for_sessions));
      }
    }

    function onSessionChange(path, value, type) {
      console.warn("Session change");
      var session_id = path.substring(path.lastIndexOf(".") + 1);
      if (type & NOTIFY_CREATE) {
        bill_for_sessions[session_id] = {};
      }

      return;
    }

    function onValueChange(path, value, type, priv_data) {
      console.log("Value " + path + " change to " + value);
      var value_name_pos = path.lastIndexOf(".");
      var node_pos = path.lastIndexOf(".", value_name_pos - 1);
      var session_pos = path.lastIndexOf(".", node_pos - 1);
      var value_name = path.substring(value_name_pos + 1);
      var session_id = path.substring(session_pos + 1, node_pos);

      bill_for_sessions[session_id][value_name] = value;
      if (priv_data !== undefined) {
        ++priv_data["value_count"];
        bill_for_sessions[session_id]["value_count"] = String(priv_data["value_count"]);
      }

      return;
    }

    function start(ctx) {
      return 0;
    }

    function stop(ctx) {
      return 0;
    }
  )";
  std::ofstream ofs2(test_js_path);
  EXPECT_TRUE(ofs2.is_open());
  ofs2.write(js.data(), js.size());
  ofs2.flush();
  ofs2.close();
  Defer { remove(test_js_path.c_str()); };

  ASSERT_TRUE(LoadConfig(config_file_path));
  MockServer server;
  auto ret = server.Init(kConfig);
  if (ret == modelbox::STATUS_NOTSUPPORT) {
    GTEST_SKIP();
  }
  ASSERT_EQ(ret, modelbox::STATUS_OK);
  ret = server.Start();
  ASSERT_EQ(ret, modelbox::STATUS_OK);
  auto stats = modelbox::Statistics::GetGlobalItem();

  // graph init
  auto flow_stats = stats->GetItem(STATISTICS_ITEM_FLOW);
  auto graph_stats = flow_stats->AddItem("demo");

  // router init
  auto msg_router = PluginMsgRouter::GetInstance();
  std::promise<bool> recv_notify;
  auto recv_handle = recv_notify.get_future();
  msg_router->RegisterRecvFunc(
      "test_router", [&recv_notify](const std::string &msg_name,
                                    const std::shared_ptr<const void> &msg_data,
                                    size_t msg_len) {
        EXPECT_EQ(msg_name, "bill");
        std::string msg_str((const char *)msg_data.get(), msg_len);
        bool parse_json_ok = false;
        try {
          auto msg_json = nlohmann::json::parse(msg_str);
          std::vector<std::string> session_name_list = {"session_id1",
                                                        "session_id2"};
          EXPECT_EQ(msg_json.size(), session_name_list.size());
          std::vector<std::vector<std::string>> expected_value = {
              {"10", "25", "1920", "1080", "20", "2"},
              {"20", "30", "1366", "768", "100", "4"}};
          for (size_t session_index = 0;
               session_index < session_name_list.size(); ++session_index) {
            auto &session_name = session_name_list[session_index];
            auto &expected_session_value = expected_value[session_index];
            auto session_json_item = msg_json.find(session_name);
            ASSERT_NE(session_json_item, msg_json.end());
            auto &session_json = session_json_item.value();
            EXPECT_EQ((std::string)session_json["video_duration"],
                      expected_session_value[0]);
            EXPECT_EQ((std::string)session_json["video_rate"],
                      expected_session_value[1]);
            EXPECT_EQ((std::string)session_json["width"],
                      expected_session_value[2]);
            EXPECT_EQ((std::string)session_json["height"],
                      expected_session_value[3]);
            EXPECT_EQ((std::string)session_json["frame_count"],
                      expected_session_value[4]);
            EXPECT_EQ((std::string)session_json["value_count"],
                      expected_session_value[5]);
          }
        } catch (std::exception &e) {
          MBLOG_ERROR << "Process json failed " << msg_str;
          MBLOG_ERROR << "Expection " << e.what();
          EXPECT_TRUE(parse_json_ok);
        }

        recv_notify.set_value(true);
      });

  // session1 begin
  {
    // modelbox task1 begin
    auto session_stats = graph_stats->AddItem("session_id1");
    auto demuxer_stats = session_stats->AddItem("demuxer");
    auto decoder_stats = session_stats->AddItem("decoder");

    // modelbox task running
    demuxer_stats->AddItem("video_duration", (int32_t)10);
    demuxer_stats->AddItem("video_rate", (int32_t)25);
    decoder_stats->AddItem("width", (int32_t)1920);
    decoder_stats->AddItem("height", (int32_t)1080);
    decoder_stats->AddItem("frame_count", (int32_t)20);

    // modelbox task end
    graph_stats->DelItem("session_id1");
  }
  // session2 begin
  {
    // modelbox task1 begin
    auto session_stats = graph_stats->AddItem("session_id2");
    auto demuxer_stats = session_stats->AddItem("demuxer");
    auto decoder_stats = session_stats->AddItem("decoder");

    // modelbox task running
    demuxer_stats->AddItem("video_duration", (int32_t)20);
    demuxer_stats->AddItem("video_rate", (int32_t)30);
    decoder_stats->AddItem("width", (int32_t)1366);
    decoder_stats->AddItem("height", (int32_t)768);
    decoder_stats->AddItem("frame_count", (int32_t)100);

    // modelbox task end
    graph_stats->DelItem("session_id2");
  }

  // modelbox exit
  flow_stats->DelItem("demo");
  EXPECT_TRUE(recv_handle.get());
  server.Stop();
}

}  // namespace modelbox
