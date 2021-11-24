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


#ifndef MODELBOX_MANAGER_H
#define MODELBOX_MANAGER_H

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/limits.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>

#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */

#define MANAGER_LOG_PATH "/var/log/modelbox/"
#define MANAGER_LOG_SIZE (1024 * 1024 * 64)
#define MANAGER_LOG_NUM (48)

extern int manager_init_server(void);

extern int manager_run(void);

extern void manager_stop(void);

extern void manager_exit(void);

#ifdef BUILD_TEST

extern void manager_force_exit(void);

#endif

#ifdef __cplusplus
}
#endif /*__cplusplus */

#endif  // !MODELBOX_MANAGER_H