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
#include "modelbox/base/status.h"

const std::string DEFAULT_FILE_REQUEST_URI = "http://127.0.0.1:8024";

namespace modelbox {

class FileGetHandler {
 public:
  /**
   * @brief get data.
   * @param buff read buffer.
   * @param size buffer size.
   * @param off current read offset.
   * @param path file path.
   * @return read result.
   */
  virtual Status Get(unsigned char *buff, size_t size, off_t off) = 0;

  /**
   * @brief get file size.
   * @param path file path.
   * @return file size.
   */
  virtual uint64_t GetFileSize() = 0;

  virtual ~FileGetHandler() = default;
};

class FileRequester {
 public:
  /*
   * @brief get FileRequester instance
   * @return FileRequester instance
   */
  static std::shared_ptr<FileRequester> GetInstance();

  Status RegisterUrlHandler(const std::string &relative_url,
                            const std::shared_ptr<FileGetHandler> &handler);

  Status DeregisterUrl(const std::string &relative_url);

  void SetMaxFileReadSize(int read_size);

  virtual ~FileRequester();

 private:
  FileRequester() = default;
  Status Init();
  void HandleFileGet(const web::http::http_request &request);
  bool IsValidRequest(const web::http::http_request &request);
  bool ReadRequestRange(const web::http::http_request &request,
                        uint64_t file_size, uint64_t &range_start,
                        uint64_t &range_end);
  void ProcessRequest(const web::http::http_request &request,
                      const std::shared_ptr<FileGetHandler> &handler,
                      uint64_t range_start, uint64_t range_end);

  static std::once_flag file_requester_init_flag_;
  std::shared_ptr<web::http::experimental::listener::http_listener> listener_;
  std::unordered_map<std::string, std::shared_ptr<FileGetHandler>>
      file_handlers_;
  std::mutex handler_lock_;
  std::shared_ptr<ThreadPool> pool_;
  int max_read_size_ = 0;
};
};  // namespace modelbox

#endif  // MODELBOX_FILE_REQUESTER_H_