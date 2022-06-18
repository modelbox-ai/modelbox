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

#include "vcn_client.h"

#include <securec.h>

#define STREAM_TYPE_MAX 3
#define RTSP_CLIENT_TYPE 1

namespace modelbox {

std::mutex VcnClient::vcn_client_lock_;

VcnClient::~VcnClient() { DeInit(); }

std::shared_ptr<VcnClient> VcnClient::GetInstance() {
  static std::shared_ptr<VcnClient> vcn_client(new VcnClient());

  std::lock_guard<std::mutex> lock(vcn_client_lock_);
  static bool is_initialized = false;
  if (true == is_initialized) {
    return vcn_client;
  }

  auto ret = vcn_client->Init();
  if (modelbox::STATUS_OK != ret.Code()) {
    MBLOG_ERROR << ret.Errormsg();
    return nullptr;
  }
  is_initialized = true;
  return vcn_client;
}

modelbox::Status VcnClient::SetSDKWrapper(
    std::shared_ptr<VcnSdkWrapper> _sdk_wrapper) {
  if (nullptr == _sdk_wrapper) {
    return {modelbox::STATUS_INVALID, "wrapper pointer is nullptr."};
  }

  if (nullptr != sdk_wrapper_) {
    if (IVS_SUCCEED != sdk_wrapper_->VcnSdkCleanup()) {
      MBLOG_WARN << "Failed to clean up Vcn SDK resource.";
    }
  }

  sdk_wrapper_ = _sdk_wrapper;
  sdk_wrapper_->VcnSdkInit();
  return modelbox::STATUS_OK;
}

modelbox::Status VcnClient::Init() {
  IVS_INT32 ret_status = IVS_SUCCEED;
  sdk_wrapper_ = std::shared_ptr<VcnSdkWrapper>(new VcnSdkWrapper);
  ret_status = sdk_wrapper_->VcnSdkInit();
  if (IVS_SUCCEED != ret_status) {
    auto err_msg = "Failed to initialize VCN SDK, error code: " +
                   std::to_string(ret_status);
    return {modelbox::STATUS_FAULT, err_msg};
  }
  return modelbox::STATUS_OK;
}

modelbox::Status VcnClient::DeInit() {
  IVS_INT32 ret_status = IVS_SUCCEED;
  ret_status = sdk_wrapper_->VcnSdkCleanup();
  if (IVS_SUCCEED != ret_status) {
    MBLOG_ERROR << "Failed to clean up VCN SDK resources, error code: " +
                       std::to_string(ret_status);
    return modelbox::STATUS_FAULT;
  }
  return modelbox::STATUS_OK;
}

modelbox::Status VcnClient::GetUrl(int32_t session_id,
                                   const std::string &camera_code,
                                   uint32_t stream_type, std::string &url) {
  std::string errmsg = "";
  if (session_id < 0 || camera_code.empty() || stream_type > STREAM_TYPE_MAX) {
    return {modelbox::STATUS_INVALID, "invalid parameters."};
  }

  IVS_URL_MEDIA_PARAM stUrlMediaPara;
  memset_s(&stUrlMediaPara, sizeof(IVS_URL_MEDIA_PARAM), 0,
           sizeof(IVS_URL_MEDIA_PARAM));
  stUrlMediaPara.ServiceType = SERVICE_TYPE_REALVIDEO;
  stUrlMediaPara.AudioDecType = AUDIO_DEC_G711U;
  stUrlMediaPara.BroadCastType = BROADCAST_UNICAST;
  stUrlMediaPara.PackProtocolType = PACK_PROTOCOL_ES;
  stUrlMediaPara.ProtocolType = PROTOCOL_RTP_OVER_TCP;
  stUrlMediaPara.TransMode = MEDIA_TRANS;
  stUrlMediaPara.VideoDecType = VIDEO_DEC_H264;
  stUrlMediaPara.iClientType = RTSP_CLIENT_TYPE;
  if (stream_type == STREAM_TYPE_SUB1) {
    stUrlMediaPara.StreamType = STREAM_TYPE_SUB1;
  } else if (stream_type == STREAM_TYPE_SUB2) {
    stUrlMediaPara.StreamType = STREAM_TYPE_SUB2;
  } else {
    stUrlMediaPara.StreamType = STREAM_TYPE_MAIN;
  }
  strncpy_s(stUrlMediaPara.stTimeSpan.cStart, IVS_TIME_LEN, " ",
            strnlen(" ", IVS_TIME_LEN - 1));
  strncpy_s(stUrlMediaPara.stTimeSpan.cEnd, IVS_TIME_LEN, " ",
            strnlen(" ", IVS_TIME_LEN - 1));

  IVS_INT32 iRet = sdk_wrapper_->VcnSdkGetUrl(session_id, camera_code.c_str(),
                                              &stUrlMediaPara, url);
  if (NeedToRetry(iRet)) {
    url = "";
    return modelbox::STATUS_AGAIN;
  } else if (IVS_SUCCEED != iRet) {
    url = "";
    return {modelbox::STATUS_FAULT, std::to_string(iRet)};
  }

  return modelbox::STATUS_OK;
}

bool VcnClient::NeedToRetry(const IVS_INT32 error_code) {
  switch (error_code) {
    case IVS_SCU_ONLINE_USER_EXPIRE:
    case IVS_SDK_RET_INVALID_SESSION_ID:
      return true;
    default:
      break;
  }

  return false;
}

modelbox::Status VcnClient::AddVcnStream(VcnInfo &info,
                                         std::shared_ptr<VcnStream> &stream) {
  std::string errmsg = "";
  const std::string errmsg_prefix = "Failed to add vcn stream: ";
  if (!IsVcnInfoValid(info)) {
    errmsg = errmsg_prefix + "invalid info.";
    return {modelbox::STATUS_INVALID, errmsg};
  }

  // check accounts record
  std::shared_ptr<VcnAccount> account = nullptr;
  std::lock_guard<std::mutex> lock(vcn_accounts_lock_);
  auto itr = std::find_if(vcn_accounts_.begin(), vcn_accounts_.end(),
                          [&info](std::shared_ptr<const VcnAccount> account_) {
                            return (account_->GetUserName() == info.user_name &&
                                    account_->GetIp() == info.ip &&
                                    account_->GetPort() == info.port &&
                                    account_->GetPassword() == info.password);
                          });

  if (vcn_accounts_.end() == itr) {
    // account not found, try to log in.
    account = std::shared_ptr<VcnAccount>(new VcnAccount(info));
    int32_t session_id;
    auto ret = LoginVcnAccount(account, session_id);
    if (modelbox::STATUS_OK != ret || session_id < SESSION_ID_MIN ||
        session_id > SESSION_ID_MAX) {
      stream = nullptr;
      errmsg = errmsg_prefix + ret.Errormsg();
      return {modelbox::STATUS_FAULT, errmsg};
    }
    account->SetSessionId(session_id);
    vcn_accounts_.emplace_back(account);
    itr = vcn_accounts_.end() - 1;
  } else if ((*itr)->GetLoginState() == false) {
    // the account has logged out, then log in again
    account = *itr;
    int32_t session_id;
    auto ret = LoginVcnAccount(account, session_id);
    if (modelbox::STATUS_OK != ret || session_id < SESSION_ID_MIN ||
        session_id > SESSION_ID_MAX) {
      stream = nullptr;
      errmsg = errmsg_prefix + ret.Errormsg();
      return {modelbox::STATUS_FAULT, errmsg};
    }
    account->SetSessionId(session_id);
  } else {
    account = *itr;
  }

  // get stream url
  auto session_id = account->GetSessionId();
  std::string url = "";
  auto ret = GetUrl(session_id, info.camera_code, info.stream_type, url);

  if (modelbox::STATUS_AGAIN == ret) {
    // user login expired, need to log in again so set the session id to -1.
    account->SetSessionId(-1);
    errmsg = errmsg_prefix + "user expired, try log in again.";
    stream = nullptr;
    return {modelbox::STATUS_AGAIN, errmsg};
  } else if (modelbox::STATUS_OK != ret) {
    // something wrong happened and can not recover
    errmsg = errmsg_prefix +
             "Failed to get stream url, user name: " + account->GetUserName() +
             ", session id: " + std::to_string(session_id) +
             ", camera code: " + info.camera_code +
             ", stream type: " + std::to_string(info.stream_type) +
             ", error code: " + ret.Errormsg();
    stream = nullptr;
    return {ret.Code(), errmsg};
  }
  MBLOG_INFO << "User name: " << info.user_name
             << ", session id: " << session_id
             << ", successfully get url: " << url;

  // add this stream record
  stream = std::shared_ptr<VcnStream>(
      new VcnStream(url, info.camera_code, session_id, account),
      [this](VcnStream *stream) {
        this->RemoveVcnStream(stream);
        delete stream;
      });
  account->AddStream();

  return modelbox::STATUS_OK;
}

modelbox::Status VcnClient::RemoveVcnStream(VcnStream *stream) {
  std::string errmsg = "";
  const std::string errmsg_prefix = "Failed to remove vcn stream: ";
  if (nullptr == stream) {
    MBLOG_ERROR << errmsg_prefix + "stream ptr is nullptr.";
    return modelbox::STATUS_INVALID;
  }

  std::lock_guard<std::mutex> lock(vcn_accounts_lock_);
  auto account = stream->GetAccount();
  account->RemoveStream();

  if (account->GetStreamsCount() > 0) {
    return modelbox::STATUS_OK;
  }

  auto ret = LogoutVcnAccount(account);
  // Even though logout failed, the inactive account would automatically logout
  // in a certain period.
  if (modelbox::STATUS_OK != ret) {
    MBLOG_WARN << errmsg_prefix + ret.Errormsg();
  }

  auto itr =
      std::find_if(vcn_accounts_.begin(), vcn_accounts_.end(),
                   [&account](std::shared_ptr<const VcnAccount> ele) {
                     return (ele->GetUserName() == account->GetUserName() &&
                             ele->GetIp() == account->GetIp() &&
                             ele->GetPort() == account->GetPort() &&
                             ele->GetPassword() == account->GetPassword());
                   });
  if (vcn_accounts_.end() == itr) {
    MBLOG_WARN << "The account to be deleted is NOT FOUND!";
    return modelbox::STATUS_NOTFOUND;
  }
  vcn_accounts_.erase(itr);
  return modelbox::STATUS_OK;
}

modelbox::Status VcnClient::LoginVcnAccount(
    std::shared_ptr<const VcnAccount> account, int32_t &session_id) {
  IVS_INT32 iRet = IVS_SUCCEED;
  std::string errmsg = "";

  IVS_LOGIN_INFO stLoginInfo = {0};
  strncpy_s(stLoginInfo.cUserName, IVS_NAME_LEN, account->GetUserName().c_str(),
            IVS_NAME_LEN - 1);
  strncpy_s(stLoginInfo.pPWD, IVS_PWD_LEN, account->GetPassword().c_str(),
            IVS_PWD_LEN - 1);
  strncpy_s(stLoginInfo.stIP.cIP, IVS_IP_LEN, account->GetIp().c_str(),
            IVS_IP_LEN - 1);
  int port = atoi(account->GetPort().c_str());
  stLoginInfo.uiPort = port;
  stLoginInfo.stIP.uiIPType = IP_V4;

  if (!strncmp(stLoginInfo.pPWD, "", IVS_PWD_LEN - 1)) {
    errmsg = "Empty password.";
    return {modelbox::STATUS_INVALID, errmsg};
  }

  MBLOG_INFO << "Ready to login vcn account, user name: "
             << stLoginInfo.cUserName << ", ip: " << stLoginInfo.stIP.cIP
             << ", port: " << stLoginInfo.uiPort;

  iRet = sdk_wrapper_->VcnSdkLogin(&stLoginInfo, &session_id);
  if (IVS_SUCCEED != iRet || session_id < SESSION_ID_MIN ||
      session_id > SESSION_ID_MAX) {
    errmsg = "Failed to login, error code: " + std::to_string(iRet);
    session_id = -1;
    return {modelbox::STATUS_FAULT, errmsg};
  }

  MBLOG_INFO << "Successfully login, session id: " << session_id;
  return modelbox::STATUS_OK;
}

modelbox::Status VcnClient::LogoutVcnAccount(
    std::shared_ptr<VcnAccount> account) {
  IVS_INT32 iRet = IVS_SUCCEED;
  std::string errmsg = "";

  if (account->GetLoginState() == false) {
    return modelbox::STATUS_OK;
  }
  IVS_INT32 session_id = account->GetSessionId();
  iRet = sdk_wrapper_->VcnSdkLogout(session_id);
  if (IVS_SUCCEED != iRet) {
    errmsg = "Failed to logout, session id: " + std::to_string(session_id) +
             ", user name: " + account->GetUserName() +
             ", error code: " + std::to_string(iRet);
    return {modelbox::STATUS_FAULT, errmsg};
  }

  MBLOG_INFO << "Successfully logout, session id: " << session_id
             << ", user name: " << account->GetUserName();
  return modelbox::STATUS_OK;
}

}  // namespace modelbox