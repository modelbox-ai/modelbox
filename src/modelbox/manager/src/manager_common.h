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


#ifndef MODELBOX_MANAGER_COMMON_HEAD_H
#define MODELBOX_MANAGER_COMMON_HEAD_H

#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */

#define HEARTBEAT_MSG 1
#define MANAGER_NAME "manager"
#define MANAGER_PID_PATH "/var/run/modelbox-manager"

#define APP_NAME_LEN 64

/* heartbeat message struct */
struct heartbeat_msg {
  long mtype;
  char name[APP_NAME_LEN];
  pid_t pid;
  time_t time;
} __attribute__((aligned));

#ifdef __cplusplus
}
#endif /*__cplusplus */

#endif  // !MODELBOX_MANAGER_COMMON_HEAD_H
