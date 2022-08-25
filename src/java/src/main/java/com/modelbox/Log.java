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

import java.text.SimpleDateFormat;
import java.util.Date;

public class Log extends NativeObject {
  enum LogLevel {
    LOG_DEBUG, LOG_INFO, LOG_NOTICE, LOG_WARN, LOG_ERROR, LOG_FATAL, LOG_OFF
  }

  public Log() {
    setNativeHandle(LogNew());
  }

  /**
   * modelbox default log append function, output log to console
   * @param level log level
   * @param file log file
   * @param lineno log file lineno
   * @param func log function
   * @param msg log message
   */
  public void print(LogLevel level, String file, int lineno, String func, String msg) {
    String timeStamp = new SimpleDateFormat("yyyy-MM-dd HH.mm.ss.SSS").format(new Date());
    System.out.printf("[%s][%s][%17s:%-4d] %s\n", timeStamp, level, file, lineno, msg);
  }

  public final void jniPrintCallback(long level, String file, int lineno, String func, String msg) {
    print(LogLevel.LOG_INFO, file, lineno, func, msg);
  }

  /**
   * Set log level
   * @param level log level
   */
  public void setLogLevel(LogLevel level) {
    LogSetLogLevel(level.ordinal());
  }

  /**
   * Get log level
   * @return level log level
   */
  public LogLevel getLogLevel() {
    return LogLevel.values()[(int)LogGetLogLevel()];
  }

  /**
   * Get current log appender
   * @return
   */
  public static Log getLogger() {
    return LogGetLogger();
  }

  /**
   * Register log appender to modelbox
   * @param log
   */
  public static void regLog(Log log) {
    LogReg(log);
  }

  /**
   * Unregister log appender, reset to default
   */
  public static void unRegLog() {
    LogUnReg();
  }

  /**
   * log debug
   * @param msg message
   */
  public static void debug(String format, Object... params) {
    printLog(LogLevel.LOG_DEBUG, format, params);
  }

  /**
   * log debug
   * @param msg message
   */
  public static void debug(String message) {
    printLog(LogLevel.LOG_DEBUG, message);
  }

  /**
   * log info
   * @param msg message
   */
  public static void info(String format, Object... params) {
    printLog(LogLevel.LOG_INFO, format, params);
  }

  /**
   * log info
   * @param msg message
   */
  public static void info(String message) {
    printLog(LogLevel.LOG_INFO, message);
  }

  /**
   * log notice
   * @param msg message
   */
  public static void notice(String format, Object... params) {
    printLog(LogLevel.LOG_NOTICE, format, params);
  }

  /**
   * log notice
   * @param msg message
   */
  public static void notice(String message) {
    printLog(LogLevel.LOG_NOTICE, message);
  }

  /**
   * log warn
   * @param msg message
   */
  public static void warn(String format, Object... params) {
    printLog(LogLevel.LOG_WARN, format, params);
  }

  /**
   * log notice
   * @param msg message
   */
  public static void warn(String message) {
    printLog(LogLevel.LOG_WARN, message);
  }

  /**
   * log error
   * @param msg message
   */
  public static void error(String format, Object... params) {
    printLog(LogLevel.LOG_ERROR, format, params);
  }

  /**
   * log error
   * @param msg message
   */
  public static void error(String message) {
    printLog(LogLevel.LOG_ERROR, message);
  }

  /**
   * log fatal
   * @param msg message
   */
  public static void fatal(String format, Object... params) {
    printLog(LogLevel.LOG_FATAL, format, params);
  }

  /**
   * log fatal
   * @param msg message
   */
  public static void fatal(String message) {
    printLog(LogLevel.LOG_FATAL, message);
  }

  private static void printLog(LogLevel level, String format, Object... params) {
    StackTraceElement stack = Thread.currentThread().getStackTrace()[3];
    String file = stack.getFileName();
    int lineno = stack.getLineNumber();
    String func = stack.getMethodName();
    LogPrint(level.ordinal(), file, lineno, func, String.format(format, params));
  }

  public native long LogNew();

  public native void LogSetLogLevel(long level);

  public native long LogGetLogLevel();

  public static native Log LogGetLogger();

  public static native void LogReg(Log log);

  public static native void LogUnReg();

  public static native void LogPrint(long level, String file, int lineno, String func, String msg);
}
