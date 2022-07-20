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

#include "modelbox/drivers/common/file_requester.h"

#include <algorithm>

#include "cpprest/producerconsumerstream.h"
#include "cpprest/uri.h"
#include "modelbox/base/log.h"

const int MAX_BLOCK_SIZE = 1 * 1024 * 1024;
const int MAX_READ_SIZE = 40;

using namespace modelbox;

FileRequester::~FileRequester() { listener_->close().wait(); }

std::once_flag FileRequester::file_requester_init_flag_;

std::shared_ptr<FileRequester> FileRequester::GetInstance() {
  static std::shared_ptr<FileRequester> server(new FileRequester());
  std::call_once(
      file_requester_init_flag_,
      [](std::shared_ptr<FileRequester> server) {
        auto ret = server->Init();
        if (modelbox::STATUS_FAULT == ret) {
          server = nullptr;
        }
      },
      server);
  return server;
}

modelbox::Status FileRequester::Init() {
  utility::string_t address = _XPLATSTR(DEFAULT_FILE_REQUEST_URI);
  web::uri_builder uri(address);
  auto addr = uri.to_uri().to_string();
  listener_ =
      std::make_shared<web::http::experimental::listener::http_listener>(addr);
  listener_->support(web::http::methods::GET,
                     [this](web::http::http_request request) {
                       this->HandleFileGet(request);
                     });
  try {
    listener_->open().wait();
    MBLOG_INFO << "File requester start to listen : " << addr;
  } catch (std::exception const &e) {
    MBLOG_ERROR << e.what();
    return modelbox::STATUS_FAULT;
  }
  pool_ = std::make_shared<modelbox::ThreadPool>(0, 8);
  pool_->SetName("File-Requester");
  return modelbox::STATUS_OK;
}

modelbox::Status FileRequester::RegisterUrlHandler(
    const std::string &relative_url,
    std::shared_ptr<modelbox::FileGetHandler> handler) {
  std::lock_guard<std::mutex> lock(handler_lock_);
  auto iter = file_handlers_.find(relative_url);
  if (iter == file_handlers_.end()) {
    file_handlers_.emplace(relative_url, handler);
    return modelbox::STATUS_OK;
  }
  MBLOG_ERROR << "Url " << relative_url << "has been registered!";
  return modelbox::STATUS_EXIST;
}

void FileRequester::SetMaxFileReadSize(int read_size) {
  if (read_size <= 0 || read_size > MAX_READ_SIZE) {
    MBLOG_ERROR << "Invalid read size, use default value for instead."
                << "Your read size:" << read_size;
    return;
  }
  max_read_size_ = read_size * MAX_BLOCK_SIZE;
  MBLOG_INFO << "Set max file read size to " << max_read_size_;
}

modelbox::Status FileRequester::DeregisterUrl(const std::string &relative_url) {
  std::lock_guard<std::mutex> lock(handler_lock_);
  auto iter = file_handlers_.find(relative_url);
  if (iter != file_handlers_.end()) {
    file_handlers_.erase(iter);
    MBLOG_INFO << "Success to deregister url: " << relative_url;
    return modelbox::STATUS_OK;
  }
  MBLOG_ERROR << "Failed to deregister url: " << relative_url
              << ", url not registered!";
  return modelbox::STATUS_NOTFOUND;
}

bool FileRequester::IsValidRequest(const web::http::http_request &request) {
  auto headers = request.headers();
  if (!headers.has("Range")) {
    MBLOG_ERROR << "Request has no header names 'Range'. Request:"
                << request.to_string();
    return false;
  }
  return true;
}

bool FileRequester::ReadRequestRange(const web::http::http_request &request,
                                     const uint64_t file_size,
                                     uint64_t &range_start,
                                     uint64_t &range_end) {
  auto headers = request.headers();
  auto range_value = headers["Range"];
  const std::string range_prefix = "bytes=";
  auto pos = range_value.find(range_prefix);
  if (pos == range_value.npos) {
    MBLOG_ERROR << "Range header has no bytes range values.";
    return false;
  }
  auto range_start_end = range_value.substr(range_prefix.size());
  auto ranges = modelbox::StringSplit(range_start_end, '-');
  if ((ranges.size() > 2) || (ranges.size() < 1)) {
    MBLOG_ERROR << "Range value is invalid."
                << "range_start_end: " << range_start_end;
    return false;
  }
  try {
    range_start = std::stoull(ranges[0]);
    if (ranges.size() == 1) {
      range_end = file_size - 1;
    } else {
      range_end = std::stoull(ranges[1]);
    }
  } catch (const std::exception &e) {
    MBLOG_ERROR << "Convert request range to int failed, range " << ranges[0]
                << ", err " << e.what();
    return false;
  }

  if ((range_start < 0) || (range_start > file_size - 1) ||
      (range_end < range_start) || (range_end > file_size - 1)) {
    MBLOG_ERROR << "Request range is invalid."
                << "Range start: " << range_start << ",range end: " << range_end
                << ", file size: " << file_size;
    return false;
  }
  if (range_end > range_start + MAX_BLOCK_SIZE) {
    range_end = std::min(
        range_end, range_start + std::max(MAX_BLOCK_SIZE, max_read_size_));
  }
  return true;
}

void FileRequester::ProcessRequest(
    web::http::http_request &request,
    std::shared_ptr<modelbox::FileGetHandler> handler, uint64_t range_start,
    uint64_t range_end) {
  uint64_t file_size = handler->GetFileSize();
  concurrency::streams::producer_consumer_buffer<unsigned char> rwbuf;
  concurrency::streams::basic_istream<uint8_t> stream(rwbuf);
  web::http::http_response response(web::http::status_codes::OK);
  response.set_body(stream);
  auto rangeResponseHeader = "bytes " + std::to_string(range_start) + "-" +
                             std::to_string(range_end) + "/" +
                             std::to_string(file_size);
  response.headers().add("Content-Range", rangeResponseHeader);
  response.headers().set_content_type(U("application/octet-stream"));
  response.headers().set_content_length((size_t)(range_end - range_start + 1));

  std::shared_ptr<unsigned char> raw_data(
      new (std::nothrow) unsigned char[MAX_BLOCK_SIZE],
      [](unsigned char *p) { delete[] p; });

  if (raw_data.get() == nullptr) {
    MBLOG_ERROR << "create raw data buffer failed.";
    request.reply(web::http::status_codes::InternalError);
    return;
  }

  auto rep = request.reply(response);
  while (range_start < range_end) {
    int read_size;
    if (range_start + MAX_BLOCK_SIZE < range_end) {
      read_size = MAX_BLOCK_SIZE;
      response.set_status_code(web::http::status_codes::PartialContent);
    } else {
      read_size = range_end - range_start + 1;
    }
    auto ret = handler->Get(raw_data.get(), read_size, range_start);
    if (modelbox::STATUS_OK != ret) {
      MBLOG_ERROR << "Get file data failed.";
      request.reply(web::http::status_codes::InternalError);
      return;
    }
    rwbuf.putn_nocopy(raw_data.get(), read_size).wait();
    rwbuf.sync().wait();
    range_start += read_size;
  }

  rwbuf.close(std::ios_base::out).wait();

  rep.wait();
}

void FileRequester::HandleFileGet(web::http::http_request request) {
  MBLOG_DEBUG << request.to_string();
  std::string path = web::http::uri::decode(request.relative_uri().path());

  std::unique_lock<std::mutex> lock_handler(handler_lock_);
  auto iter = file_handlers_.find(path);
  if (iter == file_handlers_.end()) {
    MBLOG_ERROR << "File " << path << "not found.";
    request.reply(web::http::status_codes::NotFound);
    return;
  }
  auto file_get_handler = iter->second;
  lock_handler.unlock();

  if (!IsValidRequest(request)) {
    MBLOG_ERROR << "Request for file " << path
                << " is invalid. Request: " << request.to_string();
    request.reply(web::http::status_codes::BadRequest);
    return;
  }

  int file_size = file_get_handler->GetFileSize();
  uint64_t range_start, range_end = 0;
  if (!ReadRequestRange(request, file_size, range_start, range_end)) {
    MBLOG_ERROR << "Read request range for file " << path << " filed.";
    request.reply(web::http::status_codes::BadRequest);
    return;
  }

  pool_->Submit(&FileRequester::ProcessRequest, this, request, file_get_handler,
                range_start, range_end);
  return;
}
