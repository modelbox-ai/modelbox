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

#include "modelbox/server/utils.h"

#include <list>

#include "modelbox/base/log.h"
#include "securec.h"

namespace modelbox {

IPACL::IPACL() {}
IPACL::~IPACL() {}

modelbox::Status IPACL::AddCidr(const std::string &cidr) {
  uint32_t mask_len = sizeof(uint32_t) * 8;
  std::string host;

  host = cidr;
  int pos = cidr.find_first_of('/');
  if (pos > 0) {
    auto str_mask = cidr.substr(pos + 1, cidr.length());
    mask_len = atol(str_mask.c_str());
    host = cidr.substr(0, pos);
  }

  if (host.length() == 0) {
    return {modelbox::STATUS_INVALID, "ip address is invalid"};
  }

  auto addrinfo = GetAddrInfo(host);
  if (addrinfo == nullptr) {
    return modelbox::StatusError;
  }

  if (addrinfo->ai_family == AF_INET) {
    uint32_t shift = (sizeof(uint32_t) * 8) - mask_len;
    uint32_t mask = GetIPV4Addr(addrinfo) >> shift;
    std::pair<uint32_t, uint32_t> acl(mask, shift);
    ipv4_acl_.emplace_back(acl);
    return modelbox::STATUS_OK;
  } else {
    return modelbox::STATUS_NOTSUPPORT;
  }
}

modelbox::Status IPACL::IsMatch(const std::string &ipaddr) {
  auto addrinfo = GetAddrInfo(ipaddr);
  if (addrinfo == nullptr) {
    return modelbox::StatusError;
  }

  if (addrinfo->ai_family == AF_INET) {
    uint32_t ip = GetIPV4Addr(addrinfo);
    for (const auto &mask : ipv4_acl_) {
      uint32_t ip_a = ip >> mask.second;
      uint32_t ip_b = mask.first;
      if (ip_a == ip_b || (ip_b == 0 && mask.second == sizeof(uint32_t) * 8)) {
        return modelbox::STATUS_OK;
      }
    }
  } else {
    return modelbox::STATUS_NOTSUPPORT;
  }

  return modelbox::STATUS_NOTFOUND;
}

uint32_t IPACL::GetIPV4Addr(std::shared_ptr<struct addrinfo> addrinfo) {
  auto *in4 = (struct sockaddr_in *)addrinfo->ai_addr;
  uint32_t ip = ntohl(in4->sin_addr.s_addr);
  return ip;
}

std::shared_ptr<struct addrinfo> IPACL::GetAddrInfo(const std::string &host) {
  struct addrinfo hints;
  struct addrinfo *result = nullptr;

  memset_s(&hints, sizeof(hints), 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;

  auto ret = getaddrinfo(host.c_str(), "0", &hints, &result);
  if (ret != 0) {
    modelbox::StatusError = {modelbox::STATUS_FAULT, gai_strerror(ret)};
    return nullptr;
  }

  std::shared_ptr<struct addrinfo> addrinfo(
      result, [](struct addrinfo *ptr) { freeaddrinfo(ptr); });

  return addrinfo;
}

Status SplitIPPort(const std::string host, std::string &ip, std::string &port) {
  auto pos = host.find_last_of(':');

  if (pos == std::string::npos) {
    const auto *msg = "invalid ip address, please try ip:port";
    return {STATUS_INVALID, msg};
  }

  port = host.substr(pos + 1, host.length());
  int n_port = atol(port.c_str());
  if (n_port <=0 || n_port > 65535) {
    const auto *msg = "invalid port";
    return {STATUS_INVALID, msg};
  }

  ip = host.substr(0, pos);
  /* process ipv6 format */
  pos = ip.find_first_of('[');
  if (pos != std::string::npos) {
    ip = ip.substr(pos + 1, ip.length());
  }

  pos = ip.find_first_of(']');
  if (pos != std::string::npos) {
    ip = ip.substr(0, pos);
  }

  if (ip == "") {
    ip = "0.0.0.0";
  };

  return STATUS_OK;
}

}  // namespace modelbox
