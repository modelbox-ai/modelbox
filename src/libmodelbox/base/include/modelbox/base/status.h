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


#ifndef MODELBOX_STATUS_H_
#define MODELBOX_STATUS_H_

#include <memory>
#include <ostream>

namespace modelbox {

enum StatusCode {
  STATUS_SUCCESS = 0, /* Success, Avoid using this, use STATUS_OK instead.*/
  STATUS_FAULT,       /* Fault */
  STATUS_NOTFOUND,    /* Not Found */
  STATUS_INVALID,     /* Invalid argument */
  STATUS_AGAIN,       /* Try again */
  STATUS_BADCONF,     /* Bad Config */
  STATUS_NOMEM,       /* Out of memory */
  STATUS_RANGE,       /* Out of range */
  STATUS_EXIST,       /* Already exists */
  STATUS_INTERNAL,    /* Internal error */
  STATUS_BUSY,        /* Device or resource busy */
  STATUS_PERMIT,      /* Operation not permitted */
  STATUS_NOTSUPPORT,  /* Not supported */
  STATUS_NODATA,      /* No data available */
  STATUS_NOSPACE,     /* No space left */
  STATUS_NOBUFS,      /* No buffer space available  */
  STATUS_OVERFLOW,    /* Value too large for defined data type */
  STATUS_INPROGRESS,  /* Operation now in progress */
  STATUS_ALREADY,     /* Operation already in progress */
  STATUS_TIMEDOUT,    /* Operation timed out */
  STATUS_NOSTREAM,    /* Out of streams resources */
  STATUS_RESET,       /* Request Reset by peer */
  STATUS_CONTINUE,    /* Continue operation */
  STATUS_EDQUOT,      /* Quota exceeded */
  STATUS_STOP,        /* Stop operation */
  STATUS_SHUTDOWN,    /* Shutdown operation */
  STATUS_EOF,         /* End of file */
  STATUS_NOENT,       /* No such file or directory */
  STATUS_DEADLOCK,    /* Resource deadlock */
  STATUS_NORESPONSE,  /* No response*/
  STATUS_IO,          /* Input/output error */
  STATUS_LASTFLAG,    /* Status flag, don't used it */
};

class Status {
 public:
  /**
   * @brief Status code
   */
  Status();

  /**
   * @brief Status code
   * @param status copy status from status.
   */
  Status(const Status& status);

  /**
   * @brief Status code
   * @param code create status from status code.
   */
  Status(const StatusCode& code);

  /**
   * @brief Status code
   * @param success create status from bool.
   */
  Status(const bool& success);

  /**
   * @brief Status code
   * @param code create status from code.
   * @param errmsg error mesage.
   */
  Status(const StatusCode& code, const std::string& errmsg);

  /**
   * @brief Status code
   * @param status from status.
   * @param errmsg error mesage.
   */
  Status(const Status& status, const std::string& errmsg);
  virtual ~Status();

  /**
   * @brief Make status to string
   * @return string of status.
   */
  virtual std::string ToString() const;

  /**
   * @brief Get status code.
   * @return status code.
   */
  StatusCode Code();

  /**
   * @brief Get status code in string format.
   * @return status code in string.
   */
  std::string StrCode() const;

  /**
   * @brief Get status raw code in string
   * 
   */
  std::string StrStatusCode() const;

  /**
   * @brief Set error message to status
   * @param errmsg error mesage.
   */
  void SetErrormsg(const std::string& errmsg);

  /**
   * @brief Get error message
   * @return error message
   */
  const std::string& Errormsg() const;

  /**
   * @brief Get chain error messages.
   * @return error message
   */
  std::string WrapErrormsgs() const;

  /**
   * @brief Get wrapped status.
   * @return wrapped status.
   */
  std::shared_ptr<Status> Unwrap();

  /**
   * @brief Wrap status.
   * @param status wrapped status.
   * @param code status code.
   * @param errmsg error message.
   */
  void Wrap(const Status& status, const StatusCode& code,
            const std::string& errmsg);

  /**
   * @brief Check if status equals to code
   */
  bool operator==(const StatusCode& code) const;

  /**
   * @brief Check if status equals to status
   */
  bool operator==(const Status& s) const;

  /**
   * @brief Check if status equals to bool
   */
  bool operator==(const bool& success) const;

  /**
   * @brief Check if status not equal to code
   */
  bool operator!=(const StatusCode& code) const;

  /**
   * @brief Check if status not equal to status
   */
  bool operator!=(const Status& s) const;

  /**
   * @brief Override bool function
   */
  operator bool() const;

  operator enum StatusCode() const;

 private:
  std::string WrapOnlyErrormsgs(bool with_code) const;
  std::string ErrorCodeMsgs(bool with_code) const;
  StatusCode code_ = STATUS_SUCCESS;
  std::string errmsg_;
  std::shared_ptr<Status> wrap_status_;
};

std::ostream& operator<<(std::ostream& os, const Status& s);

/**
 * @brief Status success, for performance usage
 */
extern const Status STATUS_OK;

/**
 * @brief Thread local status error like errno
 */
extern thread_local Status StatusError;

}  // namespace modelbox
#endif  // MODELBOX_STATUS_H_
