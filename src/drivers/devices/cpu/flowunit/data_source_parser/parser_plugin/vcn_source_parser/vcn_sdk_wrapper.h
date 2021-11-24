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



#ifndef MODELBOX_FLOWUNIT_VCN_SDK_WRAPPER_H_
#define MODELBOX_FLOWUNIT_VCN_SDK_WRAPPER_H_

#include <hwsdk.h>
#include <IVS_SDK.h>
#include <ivs_error.h>

#include <modelbox/base/status.h>

#define MAX_VCN_URL_LENGTH 2048
#define SESSION_ID_MIN 0
#define SESSION_ID_MAX 127

namespace modelbox {

/**
 * @brief   wrap the vcn sdk apis; help to be mocked.
 */
class VcnSdkWrapper {
  friend class VcnClient;

 private:
  virtual IVS_INT32 VcnSdkInit();
  virtual IVS_INT32 VcnSdkLogin(IVS_LOGIN_INFO *login_req_info,
                                int32_t *session_id);
  virtual IVS_INT32 VcnSdkLogout(IVS_INT32 session_id);
  virtual IVS_INT32 VcnSdkGetUrl(IVS_INT32 session_id,
                                 const IVS_CHAR *camera_code,
                                 const IVS_URL_MEDIA_PARAM *url_media_param,
                                 std::string &rtsp_url);
  virtual IVS_INT32 VcnSdkCleanup();
};


}

#endif  // MODELBOX_FLOWUNIT_VCN_SDK_WRAPPER_H_