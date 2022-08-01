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

#include "modelbox/common/log.h"
#define TLOG_MAX_LINE_LEN 4096
#include "tlog.h"

namespace modelbox {

tlog_level MblogLevelToTlogLevel(modelbox::LogLevel mblog_level) {
  tlog_level level = TLOG_INFO;
  switch (mblog_level) {
    case modelbox::LOG_DEBUG:
      level = TLOG_DEBUG;
      break;
    case modelbox::LOG_INFO:
      level = TLOG_INFO;
      break;
    case modelbox::LOG_NOTICE:
      level = TLOG_NOTICE;
      break;
    case modelbox::LOG_WARN:
      level = TLOG_WARN;
      break;
    case modelbox::LOG_ERROR:
      level = TLOG_ERROR;
      break;
    case modelbox::LOG_FATAL:
      level = TLOG_FATAL;
      break;
    case modelbox::LOG_OFF:
      level = TLOG_FATAL;
      break;
  }

  return level;
}

modelbox::LogLevel TlogLevelToMblogLevel(tlog_level tlog_level) {
  modelbox::LogLevel level = modelbox::LOG_INFO;
  switch (tlog_level) {
    case TLOG_DEBUG:
      level = modelbox::LOG_DEBUG;
      break;
    case TLOG_INFO:
      level = modelbox::LOG_INFO;
      break;
    case TLOG_NOTICE:
      level = modelbox::LOG_NOTICE;
      break;
    case TLOG_WARN:
      level = modelbox::LOG_WARN;
      break;
    case TLOG_ERROR:
      level = modelbox::LOG_ERROR;
      break;
    case TLOG_FATAL:
      level = modelbox::LOG_FATAL;
      break;
    case TLOG_END:
      level = modelbox::LOG_FATAL;
      break;
  }

  return level;
}

ModelboxServerLogger::ModelboxServerLogger() = default;

ModelboxServerLogger::~ModelboxServerLogger() {
  if (initialized_) {
    tlog_exit();
  }
}

bool ModelboxServerLogger::Init(const std::string &file, int logsize,
                                int logcount, bool logscreen) {
  if (tlog_init(file.c_str(), logsize, logcount, 0, 0) != 0) {
    return false;
  }

  if (logscreen) {
    tlog_setlogscreen(true);
  }

  initialized_ = true;
  return true;
}

void ModelboxServerLogger::SetLogLevel(modelbox::LogLevel level) {
  // set log level to tlog
  tlog_setlevel(MblogLevelToTlogLevel(level));
}

modelbox::LogLevel ModelboxServerLogger::GetLogLevel() {
  // get log level
  return TlogLevelToMblogLevel(tlog_getlevel());
}

void ModelboxServerLogger::Vprint(modelbox::LogLevel level, const char *file,
                                  int lineno, const char *func,
                                  const char *format, va_list ap) {
  tlog_vext(MblogLevelToTlogLevel(level), file, lineno, func, nullptr, format,
            ap);
}

void ModelboxServerLogger::SetVerbose(bool logscreen) {
  tlog_setlogscreen(logscreen);
}

void ModelboxServerLogger::SetLogfile(const std::string &file) {
  tlog_set_logfile(file.c_str());
}

}  // namespace modelbox