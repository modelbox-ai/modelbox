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

#include "webhook_output_broker.h"

#include <securec.h>

#include <nlohmann/json.hpp>

#include "modelbox/base/uuid.h"
#include "modelbox/device/cpu/device_cpu.h"

WebhookOutputBroker::WebhookOutputBroker() = default;
WebhookOutputBroker::~WebhookOutputBroker() = default;

modelbox::Status WebhookOutputBroker::Init(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status WebhookOutputBroker::Deinit() { return modelbox::STATUS_OK; }

std::shared_ptr<modelbox::OutputBrokerHandle> WebhookOutputBroker::Open(
    const std::shared_ptr<modelbox::Configuration> &session_config,
    const std::string &config) {
  std::string uuid;
  if (modelbox::STATUS_OK != modelbox::GetUUID(&uuid)) {
    MBLOG_ERROR << "Failed to generate a uuid for the dis output broker!";
    return nullptr;
  }

  auto handle = std::make_shared<modelbox::OutputBrokerHandle>();
  handle->broker_id_ = uuid;

  if (modelbox::STATUS_OK != ParseConfig(handle, config)) {
    MBLOG_ERROR << "Parse config to json failed";
    return nullptr;
  }

  std::unique_lock<std::mutex> guard(output_configs_lock_);
  auto iter = output_configs_.find(handle->broker_id_);
  if (iter == output_configs_.end()) {
    MBLOG_ERROR
        << "Failed to send data! Can not find the broker configuration, type: "
        << handle->output_broker_type_ << ", id: " << handle->broker_id_;
    return nullptr;
  }

  std::shared_ptr<WebhookOutputInfo> output_info = iter->second;
  utility::string_t address = U(output_info->url);
  web::http::uri uri = web::http::uri(address);

  web::http::client::http_client_config client_config;
  client_config.set_timeout(utility::seconds(30));
  client_config.set_validate_certificates(false);

  auto client = std::make_shared<web::http::client::http_client>(
      web::http::uri_builder(uri).to_uri(), client_config);
  output_clients_[handle->broker_id_] = client;

  return handle;
}

modelbox::Status WebhookOutputBroker::Write(
    const std::shared_ptr<modelbox::OutputBrokerHandle> &handle,
    const std::shared_ptr<modelbox::Buffer> &buffer) {
  if (buffer == nullptr) {
    MBLOG_ERROR << "Invalid buffer: buffer is nullptr!";
    return modelbox::STATUS_NODATA;
  }

  const auto *buffer_data = buffer->ConstData();
  if (buffer_data == nullptr) {
    MBLOG_ERROR << "Invalid buffer: buffer is nullptr!";
    return modelbox::STATUS_NODATA;
  }

  std::string data((const char *)buffer_data, buffer->GetBytes());
  if (data.empty()) {
    MBLOG_WARN << "Invalid data! Nothing to be upload!";
  }

  std::unique_lock<std::mutex> guard(output_configs_lock_);
  auto iter = output_configs_.find(handle->broker_id_);
  if (iter == output_configs_.end()) {
    MBLOG_ERROR
        << "Failed to send data! Can not find the broker configuration, type: "
        << handle->output_broker_type_ << ", id: " << handle->broker_id_;
    return modelbox::STATUS_NOTFOUND;
  }

  auto client_iter = output_clients_.find(handle->broker_id_);
  if (client_iter == output_clients_.end()) {
    MBLOG_ERROR
        << "Failed to send data! Can not find the broker clients, type: "
        << handle->output_broker_type_ << ", id: " << handle->broker_id_;
    return modelbox::STATUS_NOTFOUND;
  }

  auto client = client_iter->second;
  guard.unlock();
  std::shared_ptr<WebhookOutputInfo> output_info = iter->second;
  web::http::http_headers headers_post;
  for (auto iter = output_info->headers.begin();
       iter != output_info->headers.end(); ++iter) {
    headers_post.add(U(iter->first), U(iter->second));
  }
  web::http::http_request msg_post;
  msg_post.set_method(web::http::methods::POST);
  msg_post.headers() = headers_post;

  msg_post.set_body(data);

  try {
    web::http::http_response resp_post = client->request(msg_post).get();

    std::string msg_name;
    buffer->Get("msg_name", msg_name);

    if (resp_post.status_code() >= 200 && resp_post.status_code() < 300) {
      MBLOG_DEBUG << "Send data to webhook success. Message name: " << msg_name
                  << ". Http status code: " << resp_post.status_code()
                  << ". Response body: " << resp_post.extract_string().get();
      return modelbox::STATUS_OK;
    }

    MBLOG_WARN << "Send data to webhook failed. Message name: " << msg_name
               << ". Http status code: " << resp_post.status_code()
               << ". Response body: " << resp_post.extract_string().get()
               << ". Try again.";
    return modelbox::STATUS_AGAIN;
  } catch (std::exception const &e) {
    MBLOG_ERROR << e.what();
    return modelbox::STATUS_FAULT;
  }
}

modelbox::Status WebhookOutputBroker::Sync(
    const std::shared_ptr<modelbox::OutputBrokerHandle> &handle) {
  return modelbox::STATUS_OK;
}

modelbox::Status WebhookOutputBroker::Close(
    const std::shared_ptr<modelbox::OutputBrokerHandle> &handle) {
  std::unique_lock<std::mutex> guard(output_configs_lock_);
  auto iter = output_configs_.find(handle->broker_id_);
  if (iter == output_configs_.end()) {
    MBLOG_ERROR << "Broker handle not found, type: "
                << handle->output_broker_type_
                << ", id: " << handle->broker_id_;
    return modelbox::STATUS_NOTFOUND;
  }

  output_configs_.erase(handle->broker_id_);

  auto client_iter = output_clients_.find(handle->broker_id_);
  if (client_iter == output_clients_.end()) {
    MBLOG_ERROR << "Broker clients handle not found, type: "
                << handle->output_broker_type_
                << ", id: " << handle->broker_id_;
    return modelbox::STATUS_NOTFOUND;
  }

  output_clients_.erase(handle->broker_id_);
  guard.unlock();

  return modelbox::STATUS_OK;
}

modelbox::Status WebhookOutputBroker::ParseConfig(
    const std::shared_ptr<modelbox::OutputBrokerHandle> &handle,
    const std::string &config) {
  nlohmann::json json;
  std::shared_ptr<WebhookOutputInfo> output_info =
      std::make_shared<WebhookOutputInfo>();
  try {
    json = nlohmann::json::parse(config);
    output_info->url = json["url"].get<std::string>();
    if (output_info->url.empty()) {
      MBLOG_ERROR << "Invalid url, value of key <url> is empty!";
      return modelbox::STATUS_BADCONF;
    }
    MBLOG_DEBUG << "url: " << output_info->url;

    auto value = json["headers"].get<nlohmann::json>();
    for (const auto &header : value.items()) {
      if (header.key().empty()) {
        MBLOG_ERROR << "headers key is empty!";
        return modelbox::STATUS_BADCONF;
      }
      if (!header.value().is_string()) {
        MBLOG_ERROR << "Key <" << header.key() << "> must have string value.";
        return modelbox::STATUS_BADCONF;
      }
      output_info->headers[header.key()] = header.value();
    }

  } catch (const std::exception &e) {
    MBLOG_ERROR << "Parse output config to json failed, detail: " << e.what();
    return modelbox::STATUS_BADCONF;
  }

  MBLOG_DEBUG << "Parse cfg json success.";

  std::unique_lock<std::mutex> guard(output_configs_lock_);
  output_configs_[handle->broker_id_] = output_info;
  guard.unlock();

  return modelbox::STATUS_OK;
}