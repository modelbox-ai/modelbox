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

#ifndef MODELBOX_LOG_H_
#define MODELBOX_LOG_H_

#include <modelbox/base/utils.h>

#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace modelbox {

/**
 * @brief Log level
 */
enum LogLevel {
  /// Debug
  LOG_DEBUG = 0,
  /// Info
  LOG_INFO,
  /// Notice
  LOG_NOTICE,
  /// Warning
  LOG_WARN,
  /// Error
  LOG_ERROR,
  /// Fatal
  LOG_FATAL,
  /// Turn off log
  LOG_OFF,
};

/**
 * @brief Logger interface
 */
class Logger {
 public:
  Logger();
  virtual ~Logger();

  /**
   * @brief Output log with va-arg
   * @param level log level
   * @param file log file
   * @param lineno log file line number
   * @param func log function
   * @param format log format
   * @param ap va_list
   */
  virtual void Vprint(LogLevel level, const char *file, int lineno,
                      const char *func, const char *format, va_list ap);

  /**
   * @brief Output log
   * @param level log level
   * @param file log file
   * @param lineno log file line number
   * @param func log function
   * @param msg log message
   */
  virtual void Print(LogLevel level, const char *file, int lineno,
                     const char *func, const char *msg);
  /**
   * @brief Set log level
   * @param level log level
   */
  virtual void SetLogLevel(LogLevel level);

  /**
   * @brief Get log level
   * @return level log level
   */
  virtual LogLevel GetLogLevel() = 0;
};

using LoggerVprint =
    std::function<void(LogLevel level, const char *file, int lineno,
                       const char *func, const char *format, va_list ap)>;
using LoggerPrint =
    std::function<void(LogLevel level, const char *file, int lineno,
                       const char *func, const char *msg)>;

/**
 * @brief Register va-list log function
 * @param func va-list log function
 */
extern void RegLogVprint(LoggerVprint func);

/**
 * @brief Register print log function
 * @param func print log function
 */
extern void RegLogPrint(LoggerPrint func);

class LoggerCallback : public Logger {
 public:
  LoggerCallback();
  virtual ~LoggerCallback();

  /**
   * @brief Register va-list log function
   * @param func va-list log function
   */
  void RegVprint(LoggerVprint func);

  /**
   * @brief Register print log function
   * @param func print log function
   */
  void RegPrint(LoggerPrint func);

  /**
   * @brief Set log level
   * @param level log level
   */
  virtual void SetLogLevel(LogLevel level);

  /**
   * @brief Get log level
   * @return level log level
   */
  virtual LogLevel GetLogLevel();

 private:
  /**
   * @brief Output log with va-arg
   * @param level log level
   * @param file log file
   * @param lineno log file line number
   * @param func log function
   * @param format log format
   * @param ap va_list
   */
  virtual void Vprint(LogLevel level, const char *file, int lineno,
                      const char *func, const char *format, va_list ap);
  /**
   * @brief Output log
   * @param level log level
   * @param file log file
   * @param lineno log file line number
   * @param func log function
   * @param msg log message
   */
  virtual void Print(LogLevel level, const char *file, int lineno,
                     const char *func, const char *msg);

  LoggerVprint vprint_;
  LoggerPrint print_;
  LogLevel level_;
};

/**
 * @brief Console logger
 */
class LoggerConsole : public Logger {
 public:
  LoggerConsole();
  virtual ~LoggerConsole();

  /**
   * @brief Output log
   * @param level log level
   * @param file log file
   * @param lineno log file line number
   * @param func log function
   * @param msg log message
   */

  void Print(LogLevel level, const char *file, int lineno, const char *func,
             const char *msg);

  /**
   * @brief Set log level
   * @param level log level
   */
  void SetLogLevel(LogLevel level);

  /**
   * @brief Get log level
   * @return level log level
   */
  LogLevel GetLogLevel();

 private:
  void SetLogLevelFromEnv();
  LogLevel level_ = LOG_OFF;
};

class Log {
  using Stream = std::ostringstream;
  using Buffer_p = std::unique_ptr<Stream, std::function<void(Stream *)>>;

 public:
  Log();
  virtual ~Log();

  /**
   * @brief Output log
   * @param level log level
   * @param file log file
   * @param lineno log file line number
   * @param func log function
   * @param format log format
   */
  void Print(LogLevel level, const char *file, int lineno, const char *func,
             const char *format, ...) __attribute__((format(printf, 6, 7)))
  __attribute__((nonnull(6)));

  /**
   * @brief Output log with va-arg
   * @param level log level
   * @param file log file
   * @param lineno log file line number
   * @param func log function
   * @param format log format
   * @param ap va_list
   */
  void Vprint(LogLevel level, const char *file, int lineno, const char *func,
              const char *format, va_list ap);

  /**
   * @brief Set loggger
   * @param logger poniter to logger
   */
  void SetLogger(const std::shared_ptr<Logger> &logger);

  /**
   * @brief Whether to output log
   * @param level log level
   */
  bool CanLog(LogLevel level);

  /**
   * @brief Get loggger
   * @return logger poniter to logger
   */
  std::shared_ptr<Logger> GetLogger();

  /**
   * @brief Output log to stream
   * @param level log level
   * @param file log file
   * @param lineno log file line number
   * @param func log function
   */
  Buffer_p LogStream(LogLevel level, const char *file, int lineno,
                     const char *func);

 private:
  std::shared_ptr<Logger> logger_ = std::make_shared<LoggerConsole>();
};

class LogMessage {
 public:
  /**
   * @brief Output log message
   * @param log log pointer
   * @param level log level
   * @param file log file
   * @param lineno log file line number
   * @param func log function
   */
  LogMessage(Log *log, LogLevel level, const char *file, int lineno,
             const char *func);
  virtual ~LogMessage();

  /**
   * @brief Log stream
   * @return log stream
   */
  std::ostream &Stream();

 private:
  Log *log_;
  LogLevel level_;
  const char *file_;
  int lineno_;
  const char *func_;

  std::ostringstream msg_;
};

/**
 * @brief Global logger
 */
extern Log klogger;

/**
 * @brief Log level to string
 * @param level log level
 * @return log level in string
 */
extern const char *LogLevelToString(LogLevel level);

/**
 * @brief String log level to level
 * @param level log level in string
 * @return log level
 */
extern LogLevel LogLevelStrToLevel(const std::string &level);

}  // namespace modelbox

#ifndef BASE_FILE_NAME
#define BASE_FILE_NAME                                                     \
  (__builtin_strrchr(__FILE__, '/') ? __builtin_strrchr(__FILE__, '/') + 1 \
                                    : __FILE__)
#endif

#define ModelBoxLogger modelbox::klogger

#define MODELBOX_PRINT(level, ...) \
  ModelBoxLogger.Print(level, BASE_FILE_NAME, __LINE__, __func__, __VA_ARGS__)

#define MODELBOX_LOGSTREAM(level)                                        \
  if (ModelBoxLogger.CanLog(level))                                      \
  modelbox::LogMessage(&ModelBoxLogger, level, BASE_FILE_NAME, __LINE__, \
                       __func__)                                         \
      .Stream()

#define MODELBOX_DEBUG(...) MODELBOX_PRINT(modelbox::LOG_DEBUG, __VA_ARGS__)
#define MODELBOX_INFO(...) MODELBOX_PRINT(modelbox::LOG_INFO, __VA_ARGS__)
#define MODELBOX_NOTICE(...) MODELBOX_PRINT(modelbox::LOG_NOTICE, __VA_ARGS__)
#define MODELBOX_WARN(...) MODELBOX_PRINT(modelbox::LOG_WARN, __VA_ARGS__)
#define MODELBOX_ERROR(...) MODELBOX_PRINT(modelbox::LOG_ERROR, __VA_ARGS__)
#define MODELBOX_FATAL(...) MODELBOX_PRINT(modelbox::LOG_FATAL, __VA_ARGS__)

/// Output debug log
#define MBLOG_DEBUG MODELBOX_LOGSTREAM(modelbox::LOG_DEBUG)
/// Output info log
#define MBLOG_INFO MODELBOX_LOGSTREAM(modelbox::LOG_INFO)
// Output notice log
#define MBLOG_NOTICE MODELBOX_LOGSTREAM(modelbox::LOG_NOTICE)
/// Output warning log
#define MBLOG_WARN MODELBOX_LOGSTREAM(modelbox::LOG_WARN)
/// Output error log
#define MBLOG_ERROR MODELBOX_LOGSTREAM(modelbox::LOG_ERROR)
/// Output fatal log
#define MBLOG_FATAL MODELBOX_LOGSTREAM(modelbox::LOG_FATAL)

/**
 * @brief Print stack, level is modelbox::LOG_DEBUG|modelbox::LOG_INFO|...
 */
#define MBLOG_STACKTRACE(level) \
  MODELBOX_PRINT(level, "Stack:\n%s", modelbox::GetStackTrace().c_str())

namespace modelbox {

/**
 * @brief Abort and print stack and record log
 * @param errmsg abort message
 */
static inline void Abort(const char *errmsg) {
  MBLOG_FATAL << "Abort: " << errmsg;
  MBLOG_STACKTRACE(LOG_FATAL);
  abort();
}
}  // namespace modelbox

#endif  // MODELBOX_LOG_H_