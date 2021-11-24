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


#ifndef MODELBOX_FLOWUNIT_HTTP_UTIL_H_
#define MODELBOX_FLOWUNIT_HTTP_UTIL_H_
#include "modelbox/base/log.h"
#include "cpprest/http_listener.h"

void SafeReply(const web::http::http_request &request,
               web::http::status_code status);
void SafeReply(const web::http::http_request &request,
               web::http::status_code status, const utf8string &body_data);
void SafeReply(const web::http::http_request &request,
               web::http::status_code status,
               const concurrency::streams::istream &body_data,
               const utility::string_t &content_type);

void HandleError(pplx::task<void> &t);

utility::string_t GetSupportedMethods();

void HandleUnSupportMethod(web::http::http_request request);

class HttpRequestLimiter {
  public:
   HttpRequestLimiter(HttpRequestLimiter &&) = delete;
   HttpRequestLimiter &operator=(HttpRequestLimiter &&) = delete;
   HttpRequestLimiter(const HttpRequestLimiter &) = delete;
   HttpRequestLimiter &operator=(const HttpRequestLimiter &) = delete;

   HttpRequestLimiter() {};

   ~HttpRequestLimiter() { --request_count_; };

   static std::shared_ptr<HttpRequestLimiter> GetInstance() {
     std::lock_guard<std::mutex> lock(request_mutex_);
     if (request_count_ < max_request_) {
       ++request_count_;
       return std::make_shared<HttpRequestLimiter>();
     }

     return nullptr;
   }

   static uint64_t max_request_;
   static std::atomic_size_t request_count_;

  private:
   static std::mutex request_mutex_;
};

#endif  // MODELBOX_FLOWUNIT_HTTP_UTIL_H_