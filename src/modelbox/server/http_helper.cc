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

#include "modelbox/server/http_helper.h"

#include <modelbox/base/log.h>

#include <regex>
#include <utility>

namespace modelbox {

const HttpStatusCode HttpStatusCodes::CONTINUE = 100;
const HttpStatusCode HttpStatusCodes::SWITCHING_PROTOCOLS = 101;
const HttpStatusCode HttpStatusCodes::OK = 200;
const HttpStatusCode HttpStatusCodes::CREATED = 201;
const HttpStatusCode HttpStatusCodes::ACCEPTED = 202;
const HttpStatusCode HttpStatusCodes::NON_AUTH_INFO = 203;
const HttpStatusCode HttpStatusCodes::NO_CONTENT = 204;
const HttpStatusCode HttpStatusCodes::RESET_CONTENT = 205;
const HttpStatusCode HttpStatusCodes::PARTIAL_CONTENT = 206;
const HttpStatusCode HttpStatusCodes::MULTI_STATUS = 207;
const HttpStatusCode HttpStatusCodes::ALREADY_REPORTED = 208;
const HttpStatusCode HttpStatusCodes::IM_USED = 226;
const HttpStatusCode HttpStatusCodes::MULTIPLE_CHOICES = 300;
const HttpStatusCode HttpStatusCodes::MOVED_PERMANENTLY = 301;
const HttpStatusCode HttpStatusCodes::FOUND = 302;
const HttpStatusCode HttpStatusCodes::SEE_OTHER = 303;
const HttpStatusCode HttpStatusCodes::NOT_MODIFIED = 304;
const HttpStatusCode HttpStatusCodes::USE_PROXY = 305;
const HttpStatusCode HttpStatusCodes::TEMPORARY_REDIRECT = 307;
const HttpStatusCode HttpStatusCodes::PERMANENT_REDIRECT = 308;
const HttpStatusCode HttpStatusCodes::BAD_REQUEST = 400;
const HttpStatusCode HttpStatusCodes::UNAUTHORIZED = 401;
const HttpStatusCode HttpStatusCodes::PAYMENT_REQUIRED = 402;
const HttpStatusCode HttpStatusCodes::FORBIDDEN = 403;
const HttpStatusCode HttpStatusCodes::NOT_FOUND = 404;
const HttpStatusCode HttpStatusCodes::METHOD_NOT_ALLOWED = 405;
const HttpStatusCode HttpStatusCodes::NOT_ACCEPTABLE = 406;
const HttpStatusCode HttpStatusCodes::PROXY_AUTH_REQUIRED = 407;
const HttpStatusCode HttpStatusCodes::REQUEST_TIMEOUT = 408;
const HttpStatusCode HttpStatusCodes::CONFLICT = 409;
const HttpStatusCode HttpStatusCodes::GONE = 410;
const HttpStatusCode HttpStatusCodes::LENGTH_REQUIRED = 411;
const HttpStatusCode HttpStatusCodes::PRECONDITION_FAILED = 412;
const HttpStatusCode HttpStatusCodes::REQUEST_ENTITY_TOO_LARGE = 413;
const HttpStatusCode HttpStatusCodes::REQUEST_URI_TOO_LARGE = 414;
const HttpStatusCode HttpStatusCodes::UNSUPPORTED_MEDIA_TYPE = 415;
const HttpStatusCode HttpStatusCodes::RANGE_NOT_SATISFIABLE = 416;
const HttpStatusCode HttpStatusCodes::EXPECTATION_FAILED = 417;
const HttpStatusCode HttpStatusCodes::MISDIRECTED_REQUEST = 421;
const HttpStatusCode HttpStatusCodes::UNPROCESSABLE_ENTITY = 422;
const HttpStatusCode HttpStatusCodes::LOCKED = 423;
const HttpStatusCode HttpStatusCodes::FAILED_DEPENDENCY = 424;
const HttpStatusCode HttpStatusCodes::UPGRADE_REQUIRED = 426;
const HttpStatusCode HttpStatusCodes::PRECONDITION_REQUIRED = 428;
const HttpStatusCode HttpStatusCodes::TOO_MANY_REQUESTS = 429;
const HttpStatusCode HttpStatusCodes::REQUEST_HEADER_FIELDS_TOO_LARGE = 431;
const HttpStatusCode HttpStatusCodes::UNAVAILABLE_FOR_LEGAL_REASONS = 451;
const HttpStatusCode HttpStatusCodes::INTERNAL_ERROR = 500;
const HttpStatusCode HttpStatusCodes::NOT_IMPLEMENTED = 501;
const HttpStatusCode HttpStatusCodes::BAD_GATEWAY = 502;
const HttpStatusCode HttpStatusCodes::SERVICE_UNAVAILABLE = 503;
const HttpStatusCode HttpStatusCodes::GATEWAY_TIMEOUT = 504;
const HttpStatusCode HttpStatusCodes::HTTP_VERSION_NOT_SUPPORTED = 505;
const HttpStatusCode HttpStatusCodes::VARIANT_ALSO_NEGOTIATES = 506;
const HttpStatusCode HttpStatusCodes::INSUFFICIENT_STORAGE = 507;
const HttpStatusCode HttpStatusCodes::LOOP_DETECTED = 508;
const HttpStatusCode HttpStatusCodes::NOT_EXTENDED = 510;
const HttpStatusCode HttpStatusCodes::NETWORK_AUTHENTICATION_REQUIRED = 511;

const HttpMethod HttpMethods::GET = "GET";
const HttpMethod HttpMethods::POST = "POST";
const HttpMethod HttpMethods::PUT = "PUT";
const HttpMethod HttpMethods::DELETE = "DELETE";
const HttpMethod HttpMethods::HEAD = "HEAD";
const HttpMethod HttpMethods::OPTIONS = "OPTIONS";
const HttpMethod HttpMethods::TRACE = "TRACE";
const HttpMethod HttpMethods::CONNECT = "CONNECT";
const HttpMethod HttpMethods::MERGE = "MERGE";
const HttpMethod HttpMethods::PATCH = "PATCH";

#define GET_SSL_ERR(ssl_err_code, ssl_err_str)          \
  char ssl_err_str[256];                                \
  auto ssl_err_code_num = ERR_get_error();              \
  /* NOLINTNEXTLINE */                                  \
  auto ssl_err_code = std::to_string(ssl_err_code_num); \
  ERR_error_string_n(ssl_err_code_num, ssl_err_str, 256);

Status UseCertificate(SSL_CTX &ctx, const void *cert_buf, int len,
                      pem_password_cb cb, void *cb_user_data) {
  auto *cert_bio = BIO_new_mem_buf(cert_buf, len);
  if (cert_bio == nullptr) {
    GET_SSL_ERR(err_code, err_str);
    return {STATUS_FAULT,
            "load cert as bio failed, err: " + err_code + ", " + err_str};
  }

  auto *cert = PEM_read_bio_X509_AUX(cert_bio, nullptr, cb, cb_user_data);
  BIO_free(cert_bio);
  if (cert == nullptr) {
    GET_SSL_ERR(err_code, err_str);
    return {STATUS_FAULT,
            "read x509 failed, err: " + err_code + ", " + err_str};
  }

  auto ret = SSL_CTX_use_certificate(&ctx, cert);
  if (ret != 1) {
    GET_SSL_ERR(err_code, err_str);
    return {STATUS_FAULT, "use cert failed, err: " + err_code + ", " + err_str};
  }

  return STATUS_OK;
}

Status UsePrivateKey(SSL_CTX &ctx, const void *key_buf, int len,
                     pem_password_cb cb, void *cb_user_data) {
  auto *key_bio = BIO_new_mem_buf(key_buf, len);
  if (key_bio == nullptr) {
    GET_SSL_ERR(err_code, err_str);
    return {STATUS_FAULT,
            "load key as bio failed, err: " + err_code + ", " + err_str};
  }

  auto *key = PEM_read_bio_PrivateKey(key_bio, nullptr, cb, cb_user_data);
  BIO_free(key_bio);
  if (key == nullptr) {
    GET_SSL_ERR(err_code, err_str);
    return {STATUS_FAULT, "read key failed, err: " + err_code + ", " + err_str};
  }

  auto ret = SSL_CTX_use_PrivateKey(&ctx, key);
  if (ret != 1) {
    GET_SSL_ERR(err_code, err_str);
    return {STATUS_FAULT, "use key failed, err: " + err_code + ", " + err_str};
  }

  return STATUS_OK;
}

void HttpServerConfig::SetTimeout(const std::chrono::seconds &timeout) {
  timeout_ = timeout;
}

void HttpServerConfig::SetSSLConfigCallback(const SSLConfigCallback &cb) {
  ssl_config_cb_ = cb;
}

std::chrono::seconds HttpServerConfig::GetTimeout() const { return timeout_; }

SSLConfigCallback HttpServerConfig::GetSSLConfigCallback() const {
  return ssl_config_cb_;
}

bool HttpPathMatchNode::IsValid() { return is_valid_; }

bool HttpPathMatchNode::HasChildren() { return !children_.empty(); }

Status HttpPathMatchNode::AddChild(std::list<std::string> node_path) {
  if (node_path.empty()) {
    if (is_valid_) {
      return STATUS_EXIST;
    }

    is_valid_ = true;
    return STATUS_OK;
  }

  auto child_name = node_path.front();
  node_path.pop_front();
  std::shared_ptr<HttpPathMatchNode> child;
  auto item = children_.find(child_name);
  if (item == children_.end()) {
    child = std::make_shared<HttpPathMatchNode>();
    children_[child_name] = child;
  } else {
    child = item->second;
  }

  return child->AddChild(std::move(node_path));
}

Status HttpPathMatchNode::DelChild(std::list<std::string> node_path) {
  if (node_path.empty()) {
    is_valid_ = false;
    return STATUS_OK;
  }

  auto child_name = node_path.front();
  node_path.pop_front();
  auto item = children_.find(child_name);
  if (item == children_.end()) {
    return STATUS_NOTFOUND;
  }

  auto &child = item->second;
  auto ret = child->DelChild(std::move(node_path));
  if (!ret) {
    return ret;
  }

  if (!child->IsValid() && !child->HasChildren()) {
    children_.erase(child_name);
  }

  return STATUS_OK;
}

Status HttpPathMatchNode::Match(std::list<std::string> path,
                                std::list<std::string> &node_path) {
  auto ret = STATUS_NOTFOUND;
  if (is_valid_) {
    ret = STATUS_OK;
  }

  if (path.empty()) {
    return ret;
  }

  auto child_name = path.front();
  path.pop_front();
  auto item = children_.find(child_name);
  if (item == children_.end()) {
    return ret;
  }

  node_path.push_back(child_name);
  auto &child = item->second;
  auto child_ret = child->Match(std::move(path), node_path);
  if (child_ret) {
    return STATUS_OK;
  }

  node_path.pop_back();
  return ret;
}

Status HttpPathMatchTree::AddNode(const std::string &node_path) {
  auto node_name_list = SplitHttpPath(node_path);
  return root_.AddChild(std::move(node_name_list));
}

void HttpPathMatchTree::DelNode(const std::string &node_path) {
  auto node_name_list = SplitHttpPath(node_path);
  root_.DelChild(std::move(node_name_list));
}

Status HttpPathMatchTree::Match(const std::string &path,
                                std::string &node_path) {
  std::list<std::string> node_path_list;
  auto node_name_list = SplitHttpPath(path);
  auto ret = root_.Match(std::move(node_name_list), node_path_list);
  if (!ret) {
    return STATUS_NOTFOUND;
  }

  std::stringstream node_path_buffer;
  if (node_path_list.empty()) {
    node_path = "/";
    return STATUS_OK;
  }

  for (auto &path_name : node_path_list) {
    node_path_buffer << "/" << path_name;
  }

  node_path = node_path_buffer.str();
  return STATUS_OK;
}

std::list<std::string> HttpPathMatchTree::SplitHttpPath(
    const std::string &http_path) {
  std::list<std::string> name_list;
  auto name_start_pos = http_path.find('/') + 1;
  if (name_start_pos >= http_path.size()) {
    return name_list;
  }

  while (true) {
    auto name_end_pos = http_path.find('/', name_start_pos);
    if (name_end_pos == std::string::npos) {
      name_list.push_back(http_path.substr(name_start_pos));
      break;
    }

    name_list.push_back(
        http_path.substr(name_start_pos, name_end_pos - name_start_pos));
    name_start_pos = name_end_pos + 1;
    if (name_start_pos >= http_path.size()) {
      break;
    }
  }

  return name_list;
}

HttpServer::HttpServer(const std::string &endpoint)
    : HttpServer(endpoint, {}) {}

HttpServer::HttpServer(const std::string &endpoint,
                       const HttpServerConfig &config) {
  std::smatch http_match_result;
  std::regex http_pattern(R"((https?://)([\w\-\.]+)(:[0-9]+)?)");
  std::regex_match(endpoint, http_match_result, http_pattern);
  const size_t sub_str_count = 4;
  if (http_match_result.size() != sub_str_count) {
    status_ = {STATUS_BADCONF, endpoint + " format is wrong"};
    return;
  }

  auto scheme = http_match_result[1].str();
  auto ip = http_match_result[2].str();
  auto port_str = http_match_result[3].str();
  if (!port_str.empty()) {
    port_str = port_str.substr(1);  // remove ':'
  }

  const auto *format_tips = "http://ip[:port] or https://ip[:port]";
  if (scheme.empty()) {
    status_ = {STATUS_BADCONF,
               endpoint + " format is wrong, should be " + format_tips};
    return;
  }

  if (ip.empty()) {
    status_ = {STATUS_BADCONF,
               endpoint + " format is wrong, should be " + format_tips};
    return;
  }

  ip_ = ip;
  port_ = scheme == "http://" ? 80 : 443;
  if (!port_str.empty()) {
    port_ = std::stoi(port_str);
  }

  if (scheme == "http://") {
    server_impl_ = std::make_shared<httplib::Server>();
  } else {
    server_impl_ =
        std::make_shared<httplib::SSLServer>(config.GetSSLConfigCallback());
  }

  server_impl_->set_write_timeout(config.GetTimeout());
  server_impl_->set_read_timeout(config.GetTimeout());
  server_impl_->set_socket_options([](socket_t sock) {
    int yes = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<void *>(&yes),
               sizeof(yes));
  });
  server_impl_->set_keep_alive_max_count(1);
  if (!server_impl_->is_valid()) {
    status_ = {STATUS_FAULT, "server is not valid"};
    return;
  }
  RegisterHandleFunc();
}

HttpServer::~HttpServer() {
  Stop();
  if (server_thread_ != nullptr) {
    server_thread_->join();
  }
}

Status HttpServer::Register(const std::string &path, const HttpMethod &method,
                            const HttpHandleFunc &func) {
  std::lock_guard<std::mutex> lock(handler_data_lock_);
  auto &method_map = handler_map_[path];
  auto func_item = method_map.find(method);
  if (func_item != method_map.end()) {
    return STATUS_EXIST;
  }

  method_map[method] = func;
  match_tree_.AddNode(path);
  return STATUS_OK;
}

void HttpServer::Unregister(const std::string &path, const HttpMethod &method) {
  std::lock_guard<std::mutex> lock(handler_data_lock_);
  auto method_item = handler_map_.find(path);
  if (method_item == handler_map_.end()) {
    return;
  }

  auto &method_map = method_item->second;
  auto func_item = method_map.find(method);
  if (func_item == method_map.end()) {
    return;
  }

  method_map.erase(method);
  if (method_map.empty()) {
    handler_map_.erase(path);
    match_tree_.DelNode(path);
  }
}

Status HttpServer::Start() {
  if (status_ != STATUS_OK) {
    return status_;
  }

  std::lock_guard<std::mutex> lock(server_running_lock_);
  if (is_running_) {
    return STATUS_SUCCESS;
  }

  auto ret = server_impl_->bind_to_port(ip_.c_str(), port_);
  if (!ret) {
    status_ = {STATUS_ALREADY, "bind to " + ip_ + ":" + std::to_string(port_) +
                                   " failed, might be conflict, error " +
                                   StrError(errno)};
    return status_;
  }

  is_running_ = true;
  server_thread_ =
      std::make_shared<std::thread>(std::bind(&HttpServer::Listen, this));
  return STATUS_SUCCESS;
}

void HttpServer::Listen() {
  MBLOG_INFO << "Start listen at " << ip_ << ":" << port_;
  server_impl_->listen_after_bind();
  MBLOG_INFO << "End listen at " << ip_ << ":" << port_;
  is_running_ = false;
}

void HttpServer::Stop() {
  std::lock_guard<std::mutex> lock(server_running_lock_);
  if (!is_running_) {
    return;
  }

  server_impl_->stop();
  is_running_ = false;
}

bool HttpServer::IsRunning() { return is_running_; }

Status HttpServer::GetStatus() { return status_; }

void HttpServer::RegisterHandleFunc() {
  std::string all_path = "/.*";
  httplib::Server::Handler handle_func =
      std::bind(&HttpServer::HandleFunc, this, std::placeholders::_1,
                std::placeholders::_2);
  server_impl_->Get(all_path, handle_func);
  server_impl_->Post(all_path, handle_func);
  server_impl_->Put(all_path, handle_func);
  server_impl_->Delete(all_path, handle_func);
  server_impl_->Options(all_path, handle_func);
  server_impl_->Patch(all_path, handle_func);
}

void HttpServer::HandleFunc(const httplib::Request &request,
                            httplib::Response &response) {
  auto method = request.method;
  auto path = request.path;
  std::string handle_path;
  std::lock_guard<std::mutex> lock(handler_data_lock_);
  auto ret = match_tree_.Match(path, handle_path);
  if (!ret) {
    response.status = HttpStatusCodes::NOT_FOUND;
    return;
  }

  auto method_item = handler_map_.find(handle_path);
  if (method_item == handler_map_.end()) {
    response.status = HttpStatusCodes::NOT_FOUND;
    return;
  }

  auto &method_map = method_item->second;
  auto func_item = method_map.find(method);
  if (func_item == method_map.end()) {
    response.status = HttpStatusCodes::NOT_FOUND;
    return;
  }

  func_item->second(request, response);
}

std::unordered_map<std::string, std::shared_ptr<HttpServer>>
    HttpListener::shared_server_map_;

std::mutex HttpListener::shared_server_map_lock_;

HttpListener::HttpListener(const std::string &endpoint) : endpoint_(endpoint) {
  std::lock_guard<std::mutex> lock(shared_server_map_lock_);
  auto item = shared_server_map_.find(endpoint);
  if (item != shared_server_map_.end()) {
    shared_server_ = item->second;
    return;
  }

  shared_server_ = std::make_shared<HttpServer>(endpoint);
  shared_server_map_[endpoint] = shared_server_;
}

HttpListener::~HttpListener() {
  Stop();
  std::lock_guard<std::mutex> lock(shared_server_map_lock_);
  if (shared_server_.use_count() <= 2) {
    shared_server_map_.erase(endpoint_);
  }

  shared_server_.reset();
}

Status HttpListener::Register(const std::string &path, const HttpMethod &method,
                              const HttpHandleFunc &func) {
  auto support_func = [=](const httplib::Request &request,
                          httplib::Response &response) {
    if (enable_acl_ && !acl_.IsMatch(request.remote_addr)) {
      response.status = HttpStatusCodes::FORBIDDEN;
      response.body = "Access Denied";
      AddSafeHeader(response);
      return;
    }

    func(request, response);
  };

  shared_server_->Register(path, method, support_func);
  registered_path_method_.emplace_back(path, method);
  return STATUS_OK;
}

void HttpListener::Start() { shared_server_->Start(); }

void HttpListener::Stop() {
  for (auto &path_method : registered_path_method_) {
    shared_server_->Unregister(path_method.first, path_method.second);
  }

  registered_path_method_.clear();
}

Status HttpListener::GetStatus() { return shared_server_->GetStatus(); }

bool HttpListener::IsRunning() { return shared_server_->IsRunning(); }

void HttpListener::SetAclWhiteList(const std::vector<std::string> &white_list) {
  for (const auto &white_rule : white_list) {
    acl_.AddCidr(white_rule);
    enable_acl_ = true;
  }
}

void AddSafeHeader(httplib::Response &response) {
  response.headers.insert(std::pair<std::string, std::string>(
      "Referrer-Policy", "strict-origin-when-cross-origin"));
  response.headers.insert(std::pair<std::string, std::string>(
      "Content-Security-Policy",
      "default-src 'self'  data: 'unsafe-inline' 'unsafe-eval' "
      "console-static.huaweicloud.com res.hc-cdn.com;objectsrc 'none'; "
      "frame-ancestors 'none'"));
  response.headers.insert(
      std::pair<std::string, std::string>("X-Frame-Options", "DENY"));
}

HttpRequest::HttpRequest(std::string method, std::string url)
    : method_(std::move(method)), url_(std::move(url)) {}

void HttpRequest::SetHeaders(const httplib::Headers &headers) {
  headers_ = headers;
}

void HttpRequest::SetBody(const std::string &body) { body_ = body; }

void HttpRequest::SetResponse(const httplib::Response &response) {
  response_ = response;
}

std::string HttpRequest::GetMethod() { return method_; }

std::string HttpRequest::GetURL() { return url_; }

httplib::Headers HttpRequest::GetHeaders() { return headers_; }

std::string HttpRequest::GetRequestBody() { return body_; }

httplib::Response HttpRequest::GetResponse() { return response_; }

std::unordered_map<std::string,
                   std::function<httplib::Result(
                       httplib::Client &, const std::string &, HttpRequest &)>>
    g_httpclient_func_map = {
        {HttpMethods::GET,
         [](httplib::Client &client, const std::string &path,
            HttpRequest &request) {
           return client.Get(path.c_str(), request.GetHeaders());
         }},
        {HttpMethods::DELETE,
         [](httplib::Client &client, const std::string &path,
            HttpRequest &request) {
           return client.Delete(path.c_str(), request.GetHeaders());
         }},
        {HttpMethods::PUT,
         [](httplib::Client &client, const std::string &path,
            HttpRequest &request) {
           return client.Put(path.c_str(), request.GetHeaders(),
                             request.GetRequestBody(), nullptr);
         }},
        {HttpMethods::POST,
         [](httplib::Client &client, const std::string &path,
            HttpRequest &request) {
           return client.Post(path.c_str(), request.GetHeaders(),
                              request.GetRequestBody(), nullptr);
         }},
};

Status SendHttpRequest(HttpRequest &request) {
  auto url = request.GetURL();
  std::smatch url_match_result;
  std::regex url_pattern(R"((https?://[\w\-\.]+(:[0-9]+)?)(/.*)?)");
  auto ret = std::regex_match(url, url_match_result, url_pattern);
  if (!ret) {
    return {STATUS_BADCONF, "url " + url + " is wrong format"};
  }

  if (url_match_result.size() != 4) {
    return {STATUS_BADCONF, "url " + url + " is wrong format"};
  }

  auto scheme_host_port = url_match_result[1].str();
  auto path = url_match_result[3].str();
  if (path.empty()) {
    path = "/";
  }

  httplib::Client client(scheme_host_port);
  client.enable_server_certificate_verification(false);
  client.set_write_timeout(std::chrono::seconds(30));

  auto method = request.GetMethod();
  auto func_item = g_httpclient_func_map.find(method);
  if (func_item == g_httpclient_func_map.end()) {
    return {STATUS_NOTSUPPORT, "Not support http method " + method};
  }

  auto result = func_item->second(client, path, request);
  if (result == nullptr) {
    return {STATUS_FAULT, "Send request " + method + " failed, err " +
                              httplib::to_string(result.error())};
  }

  request.SetResponse(result.value());
  return STATUS_OK;
}

void SplitPath(const std::string &path, std::string &prefix_path,
               std::string &last_path) {
  auto last_split_start_pos = path.rfind('/');
  if (last_split_start_pos == std::string::npos) {
    prefix_path = path;
    last_path.clear();
    return;
  }

  prefix_path = path.substr(0, last_split_start_pos);
  last_path = path.substr(last_split_start_pos + 1);
}

}  // namespace modelbox