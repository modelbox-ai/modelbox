/*
 * Copyright (C) 2020 Huawei Technologies Co., Ltd. All rights reserved.
 */

#ifndef MODELBOX_MANAGER_MONITOR_CLIENT_H

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */

int app_monitor_keyfile(char *file);

int app_monitor_init(const char *name, const char *keyfile);

int app_monitor_keepalive_time(void);

int app_monitor_heartbeat_interval(void);

int app_monitor_heartbeat(void);

#ifdef __cplusplus
}
#endif /*__cplusplus */

#endif  // !MODELBOX_MANAGER_MONITOR_CLIENT_H