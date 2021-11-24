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


#include "vcn_sdk_wrapper.h"
#include "securec.h"


namespace modelbox {

IVS_INT32 VcnSdkWrapper::VcnSdkInit() { return IVS_SDK_Init(); }

IVS_INT32 VcnSdkWrapper::VcnSdkLogin(IVS_LOGIN_INFO *login_req_info,
                                     int32_t *session_id) {
  if (nullptr == login_req_info || nullptr == session_id) {
    return IVS_PARA_INVALID;
  }
  IVS_INT32 ret = IVS_SDK_Login(login_req_info, session_id);
  return ret;
}

IVS_INT32 VcnSdkWrapper::VcnSdkLogout(IVS_INT32 session_id) {
  IVS_INT32 ret = IVS_SDK_Logout(session_id);
  return ret;
}

IVS_INT32 VcnSdkWrapper::VcnSdkGetUrl(
    IVS_INT32 session_id, const IVS_CHAR *camera_code,
    const IVS_URL_MEDIA_PARAM *url_media_param, std::string &rtsp_url) {
  if (nullptr == camera_code || nullptr == url_media_param) {
    return IVS_PARA_INVALID;
  }
  IVS_CHAR url[MAX_VCN_URL_LENGTH];
  memset_s(url, MAX_VCN_URL_LENGTH, 0, MAX_VCN_URL_LENGTH);

  IVS_INT32 ret = IVS_SDK_GetRtspURL(session_id, camera_code, url_media_param,
                                     url, MAX_VCN_URL_LENGTH);
  rtsp_url = url;
  return ret;
}

IVS_INT32 VcnSdkWrapper::VcnSdkCleanup() { return IVS_SDK_Cleanup(); }

}  // namespace modelbox