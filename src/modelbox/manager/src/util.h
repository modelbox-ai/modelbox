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


#ifndef MODELBOX_MANAGER_UTIL_H
#define MODELBOX_MANAGER_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */

/* 获取系统tick值, 1ms */
unsigned long get_tick_count(void);

/* 关闭所有文件句柄，除标准输入，输出，错误 */
void close_all_fd(void);

/* 获取当前进程的路径 */
int get_prog_path(char *path, int max_len);

#ifdef __cplusplus
}
#endif /*__cplusplus */

#endif
