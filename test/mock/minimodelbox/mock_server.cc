#include "mock_server.h"

#include "../config.h"
#include "test_config.h"

namespace modelbox {

MockServer::MockServer() {}

MockServer::~MockServer() {}

std::string MockServer::GetTestGraphDir() {
  return std::string(TEST_WORKING_DIR) + "/graph";
}

std::string MockServer::GetServerURL() {
  return std::string("http://") + "0.0.0.0:11104";
}

httplib::Response MockServer::DoRequest(HttpRequest &request) {
  SendHttpRequest(request);
  return request.GetResponse();
}

void MockServer::SetDefaultConfig(std::shared_ptr<Configuration> config) {
  std::vector<std::string> plugin_path;
  plugin_path.push_back(MODELBOX_PLUGIN_SO_PATH);
  plugin_path.push_back(MODELBOX_PLUGIN_EDITOR_SO_PATH);
  if (config->GetStrings("plugin.files").size() <= 0) {
    config->SetProperty("plugin.files", plugin_path);
  }

  config->SetProperty("server.ip", "0.0.0.0");
  config->SetProperty("server.port", "11104");
  config->SetProperty("control.enable", "true");
  config->SetProperty("control.listen", CONTROL_UNIX_PATH);
  config->SetProperty("server.flow_path", MockServer::GetTestGraphDir());

  config->SetProperty("editor.enable", "true");
  config->SetProperty("editor.ip", "0.0.0.0");
  config->SetProperty("editor.port", "11104");
}

Status MockServer::Init(std::shared_ptr<Configuration> config) {
  if (config == nullptr) {
    ConfigurationBuilder builder;
    config = builder.Build();
  }

  CreateDirectory(MockServer::GetTestGraphDir());

  SetDefaultConfig(config);

  if (access(MODELBOX_PLUGIN_SO_PATH, F_OK) != 0) {
    return STATUS_NOTSUPPORT;
  }

  if (access(MODELBOX_PLUGIN_EDITOR_SO_PATH, F_OK) != 0) {
    return STATUS_NOTSUPPORT;
  }

  server_ = std::make_shared<Server>(config);
  return server_->Init();
}

Status MockServer::Start() { return server_->Start(); }

void MockServer::Stop() {
  if (server_ == nullptr) {
    return;
  }
  server_->Stop();
  server_ = nullptr;
}

}  // namespace modelbox
