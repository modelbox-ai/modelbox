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

#ifndef MODELBOX_OBS_FILE_HANDLER_H_
#define MODELBOX_OBS_FILE_HANDLER_H_

#include "modelbox/base/status.h"
#include "modelbox/drivers/common/file_requester.h"
#include "obs_client.h"

namespace modelbox {

class OBSFileHandler : public FileGetHandler {
 public:
  /**
   * @brief get data from obs.
   * @param buff read buffer.
   * @param size buffer size.
   * @param off current read offset.
   * @param path obs file path in the bucket.
   * @return read result.
   */
  modelbox::Status Get(unsigned char *buff, size_t size, off_t off) override;

  /**
   * @brief get file size from obs.
   * @param path obs file path in the bucket.
   * @return file size.
   */
  uint64_t GetFileSize() override;

  void SetOBSOption(const ObsOptions &opt);

 private:
  ObsOptions opt_;
  uint64_t file_size_ = 0;
};
};  // namespace modelbox

#endif  // MODELBOX_OBS_FILE_HANDLER_H_