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

#ifndef MODELBOX_SERVER_UTILS_H_
#define MODELBOX_SERVER_UTILS_H_

#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <vector>

#include "modelbox/base/status.h"

namespace modelbox {

/**
 * @brief IP access control list
 */
class IPACL {
 public:
  /**
   * @brief constructor
   */
  IPACL();

  /**
   * @brief destructor
   */
  virtual ~IPACL();

  /**
   * @brief Add acl in cidr format
   * @param cidr ip in cidr format
   * @return add result
   */
  modelbox::Status AddCidr(const std::string &cidr);

  /**
   * @brief check whether ip is in acl list
   * @param ipaddr ip address
   * @return STATUS_SUCCESS ip is in acl list
   *         STATUS_NOTFOUND ip is not in acl list
   */
  modelbox::Status IsMatch(const std::string &ipaddr);

 private:
  uint32_t GetIPV4Addr(const std::shared_ptr<struct addrinfo> &addrinfo);
  std::shared_ptr<struct addrinfo> GetAddrInfo(const std::string &host);

  std::vector<std::pair<uint32_t, uint32_t>> ipv4_acl_;
};

}  // namespace modelbox

#endif  // MODELBOX_SERVER_UTILS_H_
