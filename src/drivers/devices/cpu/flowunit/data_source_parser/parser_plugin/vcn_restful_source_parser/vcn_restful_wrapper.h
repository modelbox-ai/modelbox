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

#ifndef MODELBOX_FLOWUNIT_VCN_RESTFUL_WRAPPER_H_
#define MODELBOX_FLOWUNIT_VCN_RESTFUL_WRAPPER_H_

#include <modelbox/base/status.h>

#include <string>
#include <unordered_map>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

#include "vcn_info.h"

namespace modelbox {

typedef struct tag_VcnRestfulInfo : public VcnInfo {
  tag_VcnRestfulInfo(const VcnInfo &info) : VcnInfo(info), jsession_id("") {}
  tag_VcnRestfulInfo() {}
  std::string jsession_id;
} VcnRestfulInfo;

typedef enum REQ_METHOD { REQ_GET, REQ_POST } REQ_METHOD;

using HttpcliFunc = httplib::Result(httplib::Client &cli,
                                    const std::string &path,
                                    const httplib::Headers &headers,
                                    const std::string &body);

using HttpcliFuncMap =
    std::unordered_map<REQ_METHOD, std::function<HttpcliFunc>>;

/**
 * @brief   wrap the vcn restful apis; help to be mocked.
 */
class VcnRestfulWrapper {
 public:
  VcnRestfulWrapper();
  virtual ~VcnRestfulWrapper() = default;

 public:
  /**
   * @brief   restful login a vcn account
   * @param   restful_info - in, a VcnRestfulInfo object, containing information
   * to login.
   * @return  Successful or not
   */
  virtual modelbox::Status Login(VcnRestfulInfo &restful_info);

  /**
   * @brief   restful logout a vcn account
   * @param   restful_info - in, a VcnRestfulInfo object, containing
   * information to logout.
   * @return  Successful or not
   */
  virtual modelbox::Status Logout(const VcnRestfulInfo &restful_info);

  virtual modelbox::Status GetUrl(const VcnRestfulInfo &restful_info,
                                  std::string &rtsp_url);

  virtual modelbox::Status KeepAlive(const VcnRestfulInfo &restful_info);

 private:
  bool IsRestfulInfoValid(const VcnRestfulInfo &restful_info,
                          bool is_login = false);

  modelbox::Status ParseRestfulLoginResult(const httplib::Response &resp,
                                           VcnRestfulInfo &restful_info);

  modelbox::Status ParseRestfulGetUrlResult(const httplib::Response &resp,
                                            std::string &rtsp_url);

  modelbox::Status SendRequest(const std::string &uri, const std::string &path,
                               const std::string &body,
                               const httplib::Headers &headers,
                               REQ_METHOD method, httplib::Response &resp);

 private:
  HttpcliFuncMap httpcli_func_map_;
};

}  // namespace modelbox

#endif  // MODELBOX_FLOWUNIT_VCN_RESTFUL_WRAPPER_H_