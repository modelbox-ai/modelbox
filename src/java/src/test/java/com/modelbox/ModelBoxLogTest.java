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

import static org.junit.Assert.assertEquals;
import java.text.SimpleDateFormat;
import java.util.Date;
import org.junit.Test;

public class ModelBoxLogTest {

  class TestLog extends Log {
    public void print(LogLevel level, String file, int lineno, String func, String msg) {
      String timeStamp = new SimpleDateFormat("yyyy-MM-dd HH.mm.ss.SSS").format(new Date());
      System.out.printf("[%s][%s][%17s:%-4d] %s\n", timeStamp, level, file, lineno, msg);
      lastMsg = msg;
    }

    public String lastMsg;
  }

  @Test
  public void testLogReg() throws Exception {
    String mesg = "This is hello msg";
    Log.getLogger().setLogLevel(Log.LogLevel.LOG_DEBUG);
    Log.debug(mesg);
    TestLog log = new TestLog();
    Log.regLog(log);
    log.setLogLevel(Log.LogLevel.LOG_DEBUG);
    Log.info(mesg);
    assertEquals(log.lastMsg, mesg);
    assertEquals(log, Log.LogGetLogger());
    Log.unRegLog();
  }

  @Test
  public void testLogFormat() throws Exception {
    String mesg = "This is hello msg";
    String mesg1 = "This is message 2";
    String expect_msg = "Msg: " + mesg + " " + mesg1;
    Log.getLogger().setLogLevel(Log.LogLevel.LOG_DEBUG);
    Log.debug(mesg);
    TestLog log = new TestLog();
    Log.regLog(log);
    log.setLogLevel(Log.LogLevel.LOG_DEBUG);
    Log.info("Msg: %s %s", mesg, mesg1);
    assertEquals(log.lastMsg, expect_msg);
    assertEquals(log, Log.LogGetLogger());
    Log.unRegLog();
  }

  @Test
  public void testLogLevel() throws Exception {
    Log.LogLevel oldLevel = Log.getLogger().getLogLevel();

    Log.getLogger().setLogLevel(Log.LogLevel.LOG_DEBUG);
    assertEquals(Log.getLogger().getLogLevel(), Log.LogLevel.LOG_DEBUG);
    Log.getLogger().setLogLevel(Log.LogLevel.LOG_INFO);
    assertEquals(Log.getLogger().getLogLevel(), Log.LogLevel.LOG_INFO);
    Log.getLogger().setLogLevel(Log.LogLevel.LOG_NOTICE);
    assertEquals(Log.getLogger().getLogLevel(), Log.LogLevel.LOG_NOTICE);
    Log.getLogger().setLogLevel(Log.LogLevel.LOG_WARN);
    assertEquals(Log.getLogger().getLogLevel(), Log.LogLevel.LOG_WARN);
    Log.getLogger().setLogLevel(Log.LogLevel.LOG_ERROR);
    assertEquals(Log.getLogger().getLogLevel(), Log.LogLevel.LOG_ERROR);
    
    Log.getLogger().setLogLevel(oldLevel);
  }
}