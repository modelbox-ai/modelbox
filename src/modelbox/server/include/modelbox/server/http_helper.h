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

#ifndef MODELBOX_HTTP_HELPER_H_
#define MODELBOX_HTTP_HELPER_H_

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

#include <thread>
#include <unordered_map>

#include "modelbox/base/status.h"
#include "modelbox/server/utils.h"

namespace modelbox {

constexpr const char *TEXT_PLAIN = "text/plain";
constexpr const char *JSON = "application/json";

using HttpStatusCode = int32_t;

class HttpStatusCodes {
 public:
  static const HttpStatusCode CONTINUE;
  static const HttpStatusCode SWITCHING_PROTOCOLS;
  static const HttpStatusCode OK;
  static const HttpStatusCode CREATED;
  static const HttpStatusCode ACCEPTED;
  static const HttpStatusCode NON_AUTH_INFO;
  static const HttpStatusCode NO_CONTENT;
  static const HttpStatusCode RESET_CONTENT;
  static const HttpStatusCode PARTIAL_CONTENT;
  static const HttpStatusCode MULTI_STATUS;
  static const HttpStatusCode ALREADY_REPORTED;
  static const HttpStatusCode IM_USED;
  static const HttpStatusCode MULTIPLE_CHOICES;
  static const HttpStatusCode MOVED_PERMANENTLY;
  static const HttpStatusCode FOUND;
  static const HttpStatusCode SEE_OTHER;
  static const HttpStatusCode NOT_MODIFIED;
  static const HttpStatusCode USE_PROXY;
  static const HttpStatusCode TEMPORARY_REDIRECT;
  static const HttpStatusCode PERMANENT_REDIRECT;
  static const HttpStatusCode BAD_REQUEST;
  static const HttpStatusCode UNAUTHORIZED;
  static const HttpStatusCode PAYMENT_REQUIRED;
  static const HttpStatusCode FORBIDDEN;
  static const HttpStatusCode NOT_FOUND;
  static const HttpStatusCode METHOD_NOT_ALLOWED;
  static const HttpStatusCode NOT_ACCEPTABLE;
  static const HttpStatusCode PROXY_AUTH_REQUIRED;
  static const HttpStatusCode REQUEST_TIMEOUT;
  static const HttpStatusCode CONFLICT;
  static const HttpStatusCode GONE;
  static const HttpStatusCode LENGTH_REQUIRED;
  static const HttpStatusCode PRECONDITION_FAILED;
  static const HttpStatusCode REQUEST_ENTITY_TOO_LARGE;
  static const HttpStatusCode REQUEST_URI_TOO_LARGE;
  static const HttpStatusCode UNSUPPORTED_MEDIA_TYPE;
  static const HttpStatusCode RANGE_NOT_SATISFIABLE;
  static const HttpStatusCode EXPECTATION_FAILED;
  static const HttpStatusCode MISDIRECTED_REQUEST;
  static const HttpStatusCode UNPROCESSABLE_ENTITY;
  static const HttpStatusCode LOCKED;
  static const HttpStatusCode FAILED_DEPENDENCY;
  static const HttpStatusCode UPGRADE_REQUIRED;
  static const HttpStatusCode PRECONDITION_REQUIRED;
  static const HttpStatusCode TOO_MANY_REQUESTS;
  static const HttpStatusCode REQUEST_HEADER_FIELDS_TOO_LARGE;
  static const HttpStatusCode UNAVAILABLE_FOR_LEGAL_REASONS;
  static const HttpStatusCode INTERNAL_ERROR;
  static const HttpStatusCode NOT_IMPLEMENTED;
  static const HttpStatusCode BAD_GATEWAY;
  static const HttpStatusCode SERVICE_UNAVAILABLE;
  static const HttpStatusCode GATEWAY_TIMEOUT;
  static const HttpStatusCode HTTP_VERSION_NOT_SUPPORTED;
  static const HttpStatusCode VARIANT_ALSO_NEGOTIATES;
  static const HttpStatusCode INSUFFICIENT_STORAGE;
  static const HttpStatusCode LOOP_DETECTED;
  static const HttpStatusCode NOT_EXTENDED;
  static const HttpStatusCode NETWORK_AUTHENTICATION_REQUIRED;
};

using HttpMethod = std::string;

#undef DELETE

class HttpMethods {
 public:
  static const HttpMethod GET;
  static const HttpMethod POST;
  static const HttpMethod PUT;
  static const HttpMethod DELETE;
  static const HttpMethod HEAD;
  static const HttpMethod OPTIONS;
  static const HttpMethod TRACE;
  static const HttpMethod CONNECT;
  static const HttpMethod MERGE;
  static const HttpMethod PATCH;
};

using HttpHandleFunc =
    std::function<void(const httplib::Request &, httplib::Response &)>;

using SSLConfigCallback = std::function<bool(SSL_CTX &ctx)>;

Status UseCertificate(SSL_CTX &ctx, const void *cert_buf, int len,
                      pem_password_cb cb = nullptr,
                      void *cb_user_data = nullptr);

Status UsePrivateKey(SSL_CTX &ctx, const void *key_buf, int len,
                     pem_password_cb cb = nullptr,
                     void *cb_user_data = nullptr);

class HttpServerConfig {
 public:
  void SetTimeout(const std::chrono::seconds &timeout);

  void SetSSLConfigCallback(const SSLConfigCallback &cb);

  std::chrono::seconds GetTimeout() const;

  SSLConfigCallback GetSSLConfigCallback() const;

 private:
  SSLConfigCallback ssl_config_cb_{[](SSL_CTX &ctx) { return true; }};
  std::chrono::seconds timeout_{60};
};

class HttpPathMatchNode {
 public:
  bool IsValid();

  bool HasChildren();

  Status AddChild(std::list<std::string> node_path);

  Status DelChild(std::list<std::string> node_path);

  Status Match(std::list<std::string> path, std::list<std::string> &node_path);

 private:
  bool is_valid_{false};
  std::unordered_map<std::string, std::shared_ptr<HttpPathMatchNode>> children_;
};

class HttpPathMatchTree {
 public:
  Status AddNode(const std::string &node_path);

  void DelNode(const std::string &node_path);

  Status Match(const std::string &path, std::string &node_path);

 private:
  std::list<std::string> SplitHttpPath(const std::string &http_path);

  HttpPathMatchNode root_;
};

class HttpServer {
 public:
  HttpServer(const std::string &endpoint);

  HttpServer(const std::string &endpoint, const HttpServerConfig &config);

  virtual ~HttpServer();

  Status Register(const std::string &path, const HttpMethod &method,
                  const HttpHandleFunc &func);

  void Unregister(const std::string &path, const HttpMethod &method);

  void Start();

  void Stop();

  bool IsRunning();

  Status GetStatus();

 private:
  void Listen();

  void RegisterHandleFunc();

  void HandleFunc(const httplib::Request &request, httplib::Response &response);

  Status status_{STATUS_OK};
  std::shared_ptr<httplib::Server> server_impl_;
  std::string ip_;
  std::atomic_bool is_running_{false};
  int port_{0};
  std::shared_ptr<std::thread> server_thread_;

  std::mutex handler_data_lock_;
  std::unordered_map<std::string,
                     std::unordered_map<std::string, HttpHandleFunc>>
      handler_map_;
  HttpPathMatchTree match_tree_;
};

class HttpListener {
 public:
  HttpListener(const std::string &endpoint);

  virtual ~HttpListener();

  Status Register(const std::string &path, const HttpMethod &method,
                  const HttpHandleFunc &func);

  void Start();

  void Stop();

  Status GetStatus();

  bool IsRunning();

  void SetAclWhiteList(const std::vector<std::string> &white_rule);

 private:
  IPACL acl_;
  bool enable_acl_{false};
  std::string endpoint_;
  std::shared_ptr<HttpServer> shared_server_;
  std::list<std::pair<std::string, std::string>> registered_path_method_;

  static std::unordered_map<std::string, std::shared_ptr<HttpServer>>
      shared_server_map_;
  static std::mutex shared_server_map_lock_;
};

void AddSafeHeader(httplib::Response &response);

class HttpRequest {
 public:
  HttpRequest(const std::string &method, const std::string &url);

  void SetHeaders(const httplib::Headers &headers);

  void SetBody(const std::string &body);

  void SetResponse(const httplib::Response &response);

  std::string GetMethod();

  std::string GetURL();

  httplib::Headers GetHeaders();

  std::string GetRequestBody();

  httplib::Response GetResponse();

 private:
  std::string method_;
  std::string url_;
  httplib::Headers headers_;
  std::string body_;
  httplib::Response response_;
};

Status SendHttpRequest(HttpRequest &request);

void SplitPath(const std::string &path, std::string &prefix_path,
               std::string &last_path);

}  // namespace modelbox

#endif  // MODELBOX_HTTP_HELPER_H_