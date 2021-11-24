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

#ifndef MODELBOX_FILE_REQUESTER_H_
#define MODELBOX_FILE_REQUESTER_H_

#include <modelbox/base/thread_pool.h>

#include <mutex>
#include <string>
#include <unordered_map>

#include "cpprest/http_listener.h"
#include "file_handler.h"

const std::string DEFAULT_FILE_REQUEST_URI = "http://127.0.0.1:8024";

namespace modelbox {

class FileRequester {
 public:
  /*
   * @brief get FileRequester instance
   * @return FileRequester instance
   */
  static std::shared_ptr<FileRequester> GetInstance();

  modelbox::Status RegisterUrlHandler(
      const std::string &relative_url,
      std::shared_ptr<modelbox::FileGetHandler> handler);

  modelbox::Status DeregisterUrl(const std::string &relative_url);

  ~FileRequester();

 private:
  FileRequester() {}
  modelbox::Status Init();
  void HandleFileGet(web::http::http_request request);
  bool IsValidRequest(const web::http::http_request &request);
  modelbox::Status ReadRequestRange(const web::http::http_request &request,
                                    int file_size, int &range_start,
                                    int &range_end);
  void ProcessRequest(web::http::http_request &request,
                      std::shared_ptr<modelbox::FileGetHandler> handler,
                      int range_start, int range_end);

 private:
  static std::once_flag file_requester_init_flag_;
  std::shared_ptr<web::http::experimental::listener::http_listener> listener_;
  std::unordered_map<std::string, std::shared_ptr<modelbox::FileGetHandler>>
      file_handlers_;
  std::mutex handler_lock_;
  std::shared_ptr<modelbox::ThreadPool> pool_;
};
};  // namespace modelbox

#endif  // MODELBOX_FILE_REQUESTER_H_