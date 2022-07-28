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

#include "modelbox/base/log.h"

#include <libgen.h>
#include <stdarg.h>
#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include "securec.h"

namespace modelbox {

constexpr int LOG_BUFF_SIZE = 4096;

thread_local const char *kLogID;
Log klogger;
std::shared_ptr<LoggerCallback> kloggercallback =
    std::make_shared<LoggerCallback>();

Log &GetLogger() { return klogger; }

void LogIdReset(const void *id) {
  if (id == kLogID) {
    kLogID = nullptr;
  }
}

extern std::shared_ptr<const void> LogSetLogID(const char *id) {
  kLogID = id;
  std::shared_ptr<const void> ret(id, LogIdReset);
  return ret;
}

const char *kLogLevelString[] = {
    "DEBUG", "INFO", "NOTICE", "WARN", "ERROR", "FATAL", "OFF",
};

const char *LogLevelToString(LogLevel level) {
  if (level >= LOG_OFF) {
    return "";
  }

  return kLogLevelString[level];
}

LogLevel LogLevelStrToLevel(const std::string &level) {
  StatusError = STATUS_OK;
  auto uppercase_level = level;
  std::transform(uppercase_level.begin(), uppercase_level.end(),
                 uppercase_level.begin(), ::toupper);
  static int level_num = sizeof(kLogLevelString) / sizeof(char *);
  for (int i = 0; i < level_num; i++) {
    if (uppercase_level == kLogLevelString[i]) {
      return LogLevel(i);
    }
  }

  StatusError = {STATUS_BADCONF, "config level is invalid."};
  return LOG_OFF;
}

Logger::Logger(){};
Logger::~Logger(){};

void Logger::Vprint(LogLevel level, const char *file, int lineno,
                    const char *func, const char *format, va_list ap) {
  char buff[LOG_BUFF_SIZE];

  va_list tmp;
  va_copy(tmp, ap);
  Defer { va_end(tmp); };

  auto ret = vsnprintf_s(buff, sizeof(buff), sizeof(buff) - 1, format, ap);
  if (ret < 0) {
    int huge_buff_size = LOG_BUFF_SIZE * 8;
    auto *huge_buff = (char *)malloc(huge_buff_size);
    if (huge_buff == nullptr) {
      return;
    }

    Defer {
      free(huge_buff);
      huge_buff = nullptr;
    };

    ret =
        vsnprintf_s(huge_buff, huge_buff_size, huge_buff_size - 1, format, tmp);
    if (ret < 0) {
      return;
    }

    huge_buff[huge_buff_size - 1] = '\0';
    Print(level, file, lineno, func, huge_buff);
    return;
  }

  buff[LOG_BUFF_SIZE - 1] = '\0';
  Print(level, file, lineno, func, buff);
}

void Logger::Print(LogLevel level, const char *file, int lineno,
                   const char *func, const char *msg) {}

void Logger::SetLogLevel(LogLevel level) { UNUSED_VAR(level); };

LoggerCallback::LoggerCallback() : level_(LOG_DEBUG){};

LoggerCallback::~LoggerCallback(){};

void LoggerCallback::SetLogLevel(LogLevel level) { level_ = level; }

LogLevel LoggerCallback::GetLogLevel() { return level_; };

void LoggerCallback::RegVprint(LoggerVprint func) { vprint_ = func; };

void LoggerCallback::RegPrint(LoggerPrint func) { print_ = func; };

void RegLogVprint(LoggerVprint func) {
  ModelBoxLogger.SetLogger(kloggercallback);
  kloggercallback->RegVprint(func);
}

void RegLogPrint(LoggerPrint func) {
  ModelBoxLogger.SetLogger(kloggercallback);
  kloggercallback->RegPrint(func);
}

void LoggerCallback::Vprint(LogLevel level, const char *file, int lineno,
                            const char *func, const char *format, va_list ap) {
  if (vprint_) {
    vprint_(level, file, lineno, func, format, ap);
    return;
  }

  Logger::Vprint(level, file, lineno, func, format, ap);
}

void LoggerCallback::Print(LogLevel level, const char *file, int lineno,
                           const char *func, const char *msg) {
  if (print_) {
    print_(level, file, lineno, func, msg);
    return;
  }

  Logger::Print(level, file, lineno, func, msg);
}

LoggerConsole::LoggerConsole() { SetLogLevelFromEnv(); }

void LoggerConsole::SetLogLevelFromEnv() {
  const char *log_level = getenv("MODELBOX_CONSOLE_LOGLEVEL");
  if (log_level == nullptr) {
    return;
  }

  level_ = LogLevelStrToLevel(log_level);
}

LoggerConsole::~LoggerConsole() {}

void LoggerConsole::Print(LogLevel level, const char *file, int lineno,
                          const char *func, const char *msg) {
  UNUSED_VAR(func);
  if (level_ > level) {
    return;
  }

  auto now_clock = std::chrono::system_clock::now();
  std::time_t now = std::chrono::system_clock::to_time_t(now_clock);
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now_clock.time_since_epoch()) %
                1000;

  constexpr int PREFIX_BUFF_LEN = 128;
  char prefix_msg[PREFIX_BUFF_LEN];
  char filename[PREFIX_BUFF_LEN];

  struct tm *local_tm = std::localtime(&now);
  std::string s(30, '\0');
  if (local_tm) {
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", local_tm);
  }

  int prefix_len =
      snprintf_s(prefix_msg, sizeof(prefix_msg), sizeof(prefix_msg) - 1,
                 "[%s.%.3ld][%5s][%17s:%-4d] ", s.c_str(), millis.count(),
                 LogLevelToString(level), file, lineno);

  if (prefix_len >= (int)sizeof(prefix_msg)) {
    strncpy_s(filename, sizeof(filename), file, PREFIX_BUFF_LEN - 1);
    prefix_len =
        snprintf_s(prefix_msg, sizeof(prefix_msg), sizeof(prefix_msg) - 1,
                   "[%s.%.3ld][%5s][%17s:%-4d] ", s.c_str(), millis.count(),
                   LogLevelToString(level), basename(filename), lineno);
    if (prefix_len >= (int)sizeof(prefix_msg)) {
      printf("[%s.%.3ld][%5s][?] %s\n", s.c_str(), millis.count(),
             LogLevelToString(level), msg);
      return;
    }
  }

  printf("%s%s\n", prefix_msg, msg);
}

void LoggerConsole::SetLogLevel(LogLevel level) { level_ = level; }

LogLevel LoggerConsole::GetLogLevel() { return level_; }

Log::Log() {}

Log::~Log() {}

void Log::Print(LogLevel level, const char *file, int lineno, const char *func,
                const char *format, ...) {
  if (CanLog(level) == false) {
    return;
  }

  va_list ap;
  va_start(ap, format);
  logger_->Vprint(level, file, lineno, func, format, ap);
  va_end(ap);
}

void Log::Vprint(LogLevel level, const char *file, int lineno, const char *func,
                 const char *format, va_list ap) {
  logger_->Vprint(level, file, lineno, func, format, ap);
}

bool Log::CanLog(LogLevel level) {
  if (level < logger_->GetLogLevel()) {
    return false;
  }

  return true;
}

Log::Buffer_p Log::LogStream(LogLevel level, const char *file, int lineno,
                             const char *func) {
  return Buffer_p(new Stream, [=](Stream *st) {
    Print(level, file, lineno, func, "%s", st->str().c_str());
    delete st;
  });
}

void Log::SetLogger(const std::shared_ptr<Logger> &logger) {
  if (logger == nullptr) {
    logger_ = std::make_shared<LoggerConsole>();
    return;
  }

  logger_ = logger;
}

std::shared_ptr<Logger> Log::GetLogger() { return logger_; }

LogMessage::LogMessage(Log *log, LogLevel level, const char *file, int lineno,
                       const char *func) {
  log_ = log;
  level_ = level;
  file_ = file;
  lineno_ = lineno;
  func_ = func;
}

LogMessage::~LogMessage() {
  if (kLogID) {
    log_->Print(level_, file_, lineno_, func_, "[%s] %s", kLogID,
                msg_.str().c_str());
  } else {
    log_->Print(level_, file_, lineno_, func_, "%s", msg_.str().c_str());
  }
}

std::ostream &LogMessage::Stream() { return msg_; }

}  // namespace modelbox
