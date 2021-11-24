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


#ifndef MODELBOX_FLOWUNIT_VCN_SDK_WRAPPER_MOCK_H_
#define MODELBOX_FLOWUNIT_VCN_SDK_WRAPPER_MOCK_H_

#include <IVS_SDK.h>
#include <modelbox/base/status.h>
#include <hwsdk.h>
#include <ivs_error.h>
#include "vcn_sdk_wrapper.h"

namespace modelbox {

/**
 * @brief   wrap the vcn sdk apis; help to be mocked.
 */
class VcnSdkWrapperMock : public VcnSdkWrapper {
  friend class VcnClient;

 private:
  IVS_INT32 VcnSdkInit() override;
  IVS_INT32 VcnSdkLogin(IVS_LOGIN_INFO *login_req_info,
                        int32_t *session_id) override;
  IVS_INT32 VcnSdkLogout(IVS_INT32 session_id) override;
  IVS_INT32 VcnSdkGetUrl(IVS_INT32 session_id, const IVS_CHAR *camera_code,
                         const IVS_URL_MEDIA_PARAM *url_media_param,
                         std::string &rtsp_url) override;
  IVS_INT32 VcnSdkCleanup() override;
};

}  // namespace modelbox

#endif  // MODELBOX_FLOWUNIT_VCN_SDK_WRAPPER_MOCK_H_