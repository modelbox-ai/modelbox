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

#ifndef MODELBOX_COMMON_LOG_H_
#define MODELBOX_COMMON_LOG_H_

#include <modelbox/base/log.h>

namespace modelbox {

class ModelboxServerLogger : public modelbox::Logger {
 public:
  ModelboxServerLogger();
  ~ModelboxServerLogger() override;

  /**
   * @brief Init server log
   * @param file path of logging file.
   * @param logsize max log file size.
   * @param logcount max log file number.
   * @param logscreen enable output log to screen.
   * @return init result.
   */
  bool Init(const std::string &file, int logsize, int logcount, bool logscreen);

  /**
   * @brief Output log with va-arg
   * @param level log level
   * @param file log file
   * @param lineno log file line number
   * @param func log function
   * @param format log format
   * @param ap va_list
   */
  void Vprint(modelbox::LogLevel level, const char *file, int lineno,
              const char *func, const char *format, va_list ap) override;

  /**
   * @brief Set log level
   * @param level log level
   */
  void SetLogLevel(modelbox::LogLevel level) override;

  /**
   * @brief Get log level
   * @return level log level
   */
  modelbox::LogLevel GetLogLevel() override;

  /**
   * @brief Enable or disable log to screen.
   * @param logscreen enable or disable flag.
   */
  void SetVerbose(bool logscreen);

  /**
   * @brief Change log file path.
   * @param file new log file path.
   */
  void SetLogfile(const std::string &file);

 private:
  bool initialized_{false};
};

}  // namespace modelbox

#endif  // MODELBOX_COMMON_LOG_H_