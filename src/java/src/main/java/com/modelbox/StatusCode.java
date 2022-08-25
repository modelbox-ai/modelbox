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


package com.modelbox;

public enum StatusCode {
  STATUS_SUCCESS,     /* Success, Avoid using this, use STATUS_OK instead.*/
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
  STATUS_IO           /* Input/output error */
}
