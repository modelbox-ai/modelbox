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

#include "modelbox/base/status.h"

#include <stdio.h>

#include <sstream>

namespace modelbox {

const Status STATUS_OK = STATUS_SUCCESS;
thread_local Status StatusError;

const char* kStatusCodeString[] = {
    "Success",
    "Fault",
    "Not found",
    "Invalid argument",
    "Try again",
    "Bad config",
    "Out of memory",
    "Out of range",
    "Already exists",
    "Internal error",
    "Device or resource busy",
    "Operation not permitted",
    "Not supported",
    "No data available",
    "No space left",
    "No buffer space available",
    "Value too large for defined data type",
    "Operation now in progress",
    "Operation already in progress",
    "Operation timed out",
    "Out of streams resources",
    "Request reset",
    "Continue operation",
    "Quota exceeded",
    "Stop operation",
    "Shutdown operation",
    "End of file",
    "No such file or directory",
    "Resource deadlock",
    "No response",
    "Input/output error",
    "End flag",
};

const char* kStatusCodeRawString[] = {
    "STATUS_SUCCESS",    "STATUS_FAULT",    "STATUS_NOTFOUND",
    "STATUS_INVALID",    "STATUS_AGAIN",    "STATUS_BADCONF",
    "STATUS_NOMEM",      "STATUS_RANGE",    "STATUS_EXIST",
    "STATUS_INTERNAL",   "STATUS_BUSY",     "STATUS_PERMIT",
    "STATUS_NOTSUPPORT", "STATUS_NODATA",   "STATUS_NOSPACE",
    "STATUS_NOBUFS",     "STATUS_OVERFLOW", "STATUS_INPROGRESS",
    "STATUS_ALREADY",    "STATUS_TIMEDOUT", "STATUS_NOSTREAM",
    "STATUS_RESET",      "STATUS_CONTINUE", "STATUS_EDQUOT",
    "STATUS_STOP",       "STATUS_SHUTDOWN", "STATUS_EOF",
    "STATUS_NOENT",      "STATUS_DEADLOCK", "STATUS_NORESPONSE",
    "STATUS_IO",
};

Status::Status() = default;

Status::~Status() = default;

Status::Status(const StatusCode& code) { code_ = code; }

Status::Status(const bool& success) {
  if (success) {
    code_ = STATUS_SUCCESS;
  } else {
    code_ = STATUS_FAULT;
  }
}

Status::Status(const Status& status) {
  code_ = status.code_;
  operator=(status);
}

Status::Status(const StatusCode& code, const std::string& errmsg) {
  code_ = code;
  errmsg_ = errmsg;
}

Status::Status(const Status& status, const std::string& errmsg) {
  Wrap(status, status.code_, errmsg);
}

void Status::Wrap(const Status& status, const StatusCode& code,
                  const std::string& errmsg) {
  if (code >= STATUS_LASTFLAG) {
    return;
  }

  code_ = code;
  errmsg_ = errmsg;
  wrap_status_ = std::make_shared<Status>(status);
}

std::shared_ptr<Status> Status::Unwrap() { return wrap_status_; }

StatusCode Status::Code() { return code_; }

bool Status::operator==(const StatusCode& code) const { return code_ == code; }

bool Status::operator==(const Status& s) const { return code_ == s.code_; }

bool Status::operator==(const bool& success) const {
  if ((success && code_ == STATUS_SUCCESS) ||
      (!success && code_ != STATUS_SUCCESS)) {
    return true;
  }

  return false;
}

bool Status::operator!=(const StatusCode& code) const { return code_ != code; }

bool Status::operator!=(const Status& s) const { return code_ != s.code_; }

Status::operator bool() const { return code_ == STATUS_SUCCESS; }

Status::operator enum StatusCode() const { return code_; }

std::string Status::ToString() const {
  if (errmsg_.length() > 0) {
    std::ostringstream oss;
    oss << "code: " << StrCode() << ", errmsg: " << errmsg_;
    return oss.str();
  }

  return StrCode();
}

std::string Status::StrCode() const {
  if ((size_t)code_ >= sizeof(kStatusCodeString) / sizeof(char*)) {
    return "";
  }

  return kStatusCodeString[code_];
}

std::string Status::StrStatusCode() const {
  if ((size_t)code_ >= sizeof(kStatusCodeString) / sizeof(char*)) {
    return "";
  }

  return kStatusCodeRawString[code_];
}

void Status::SetErrormsg(const std::string& errmsg) { errmsg_ = errmsg; }

const std::string& Status::Errormsg() const { return errmsg_; }

std::string Status::ErrorCodeMsgs(bool with_code) const {
  if (with_code) {
    if (Errormsg().length() > 0) {
      return StrCode() + ", " + Errormsg();
    }

    return StrCode();
  }

  return Errormsg();
}

std::string Status::WrapOnlyErrormsgs(bool with_code) const {
  if (wrap_status_ == nullptr) {
    return ErrorCodeMsgs(false);
  }

  if (Errormsg().length() == 0 && with_code == false) {
    return wrap_status_->WrapOnlyErrormsgs(with_code);
  }

  const auto& msg = wrap_status_->WrapOnlyErrormsgs(with_code);
  if (msg.length() > 0) {
    return ErrorCodeMsgs(with_code) + " -> " + msg;
  }

  return ErrorCodeMsgs(with_code);
}

std::string Status::WrapErrormsgs() const {
  if (wrap_status_ != nullptr) {
    auto msg = wrap_status_->WrapOnlyErrormsgs(false);
    if (msg.length() > 0) {
      return ErrorCodeMsgs(true) + " -> " + msg;
    }

    return ErrorCodeMsgs(true);
  }

  return ErrorCodeMsgs(true);
}

std::ostream& operator<<(std::ostream& os, const Status& s) {
  os << s.ToString();
  return os;
}

}  // namespace modelbox