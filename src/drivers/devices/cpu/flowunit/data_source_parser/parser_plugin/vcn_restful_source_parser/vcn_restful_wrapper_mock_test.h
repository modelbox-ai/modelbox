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

#ifndef MODELBOX_FLOWUNIT_VCN_RESTFUL_WRAPPER_MOCK_H_
#define MODELBOX_FLOWUNIT_VCN_RESTFUL_WRAPPER_MOCK_H_

#include <modelbox/base/status.h>

#include "vcn_restful_wrapper.h"

namespace modelbox {

/**
 * @brief   wrap the vcn sdk apis; help to be mocked.
 */
class VcnRestfulWrapperMock : public VcnRestfulWrapper {
 public:
  modelbox::Status Login(VcnRestfulInfo &restful_info) override;
  modelbox::Status Logout(const VcnRestfulInfo &restful_info) override;
  modelbox::Status GetUrl(const VcnRestfulInfo &restful_info,
                          std::string &rtsp_url) override;
  modelbox::Status KeepAlive(const VcnRestfulInfo &restful_info) override;
};

}  // namespace modelbox

#endif  // MODELBOX_FLOWUNIT_VCN_RESTFUL_WRAPPER_MOCK_H_