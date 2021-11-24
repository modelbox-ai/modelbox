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


public class Log {
  enum LogLevel {
    LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_FATAL
  }
  
  private long logPtr = 0;

  public Log() {
    logPtr = ModelBoxJni.LogNew();
  }

  protected void finalize() {
    ModelBoxJni.LogFree(logPtr);
  }

  public void print(LogLevel level, String file, int lineno, String func, String msg) {
    String timeStamp = new SimpleDateFormat("yyyy-MM-dd HH.mm.ss.SSS").format(new Date());
    System.out.printf("[%s][%s][%17s:%-4d] %s\n", timeStamp, level, file, lineno, msg);
  }

  public final void jniPrintCallback(long level, String file, int lineno, String func, String msg) {
    print(LogLevel.LOG_INFO, file, lineno, func, msg);
  }

  long getLogPtr() {
    return logPtr;
  }

  void setLogLevel(LogLevel level) {
    ModelBoxJni.SetLogLevel(logPtr, level.ordinal());
  }

  public static void regLog(Log log) {
    ModelBoxJni.LogReg(log);
  }

  public static void unRegLog() {
    ModelBoxJni.LogUnReg();
  }

  public static void debug(String msg) {
    printLog(LogLevel.LOG_DEBUG, msg);
  }

  public static void info(String msg) {

    printLog(LogLevel.LOG_INFO, msg);
  }

  public static void warn(String msg) {
    printLog(LogLevel.LOG_WARN, msg);
  }

  public static void error(String msg) {
    printLog(LogLevel.LOG_ERROR, msg);
  }

  public static void fatal(String msg) {
    printLog(LogLevel.LOG_FATAL, msg);
  }

  private static void printLog(LogLevel level, String msg) {
    StackTraceElement stack = Thread.currentThread().getStackTrace()[3];
    String file = stack.getFileName();
    int lineno = stack.getLineNumber();
    String func = stack.getMethodName();
    ModelBoxJni.LogPrint(level.ordinal(), file, lineno, func, msg);
  }
}
