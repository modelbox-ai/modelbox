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

#include "modelbox/base/uuid.h"

#include "modelbox/base/utils.h"

#define UUID_GENERATION_PATH "/proc/sys/kernel/random/uuid"

namespace modelbox {

Status GetUUID(std::string* uuid) {
  char tmp[UUID_LENGTH];
  FILE* fd = fopen(UUID_GENERATION_PATH, "r");
  if (fd == nullptr) {
    return STATUS_FAULT;
  }
  Defer { fclose(fd); };

  size_t result = fread(tmp, 1, UUID_LENGTH - 1, fd);
  if (result != UUID_LENGTH - 1) {
    return STATUS_FAULT;
  }

  tmp[UUID_LENGTH - 1] = '\0';
  *uuid = std::string(tmp);
  return STATUS_OK;
}
}  // namespace modelbox