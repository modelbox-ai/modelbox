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

#ifndef MODELBOX_FLOWUNIT_OBS_CLIENT_H_
#define MODELBOX_FLOWUNIT_OBS_CLIENT_H_

#include <modelbox/base/log.h>
#include <modelbox/base/status.h>

#include <mutex>
#include <vector>

#include "eSDKOBS.h"

namespace modelbox {

typedef struct tag_ObsOptions {
  std::string end_point;
  std::string bucket;
  std::string path;
  std::string domain_name;
  std::string xrole_name;
  std::string user_id;
  std::string ak;
  std::string sk;
  std::string token;
} ObsOptions;

/**
 * @brief This is a singleton class, in charge of all about the OBS SDK.
 */
class ObsClient {
 public:
  /**
   * @brief   Get an ObsClient object.
   * @return  Pointer to an ObsClient object.
   *          Notes: 1. return nullptr if it's failed to new the object;
   *                 2. return nullptr if it's failed to initialize the OBS SDK
   */
  static std::shared_ptr<ObsClient> GetInstance();

  /**
   * @brief   List the objects in a certain OBS path.
   * @param   opt - in, OBS options.
   * @param   object_list - out, objects list vector.
   * @return  modelbox::STATUS_OK - Successfully get the list.
   *          other status - Failed.
   */
  modelbox::Status GetObjectsList(const ObsOptions &opt,
                                  std::vector<std::string> &object_list);

  /**
   * @brief   Get an object from OBS.
   * @param   opt - in, OBS options.
   * @param   file_local_path - in, the object would be downloaded to this path.
   *          Notes: 1. directories would be made recursively if those do not
   * exist.
   *                 2. Read-Write right is needed to access the path.
   * @return  modelbox::STATUS_OK - Successfully get the list.
   *          modelbox::STATUS_AGAIN - Need to try again.
   *          other status - Failed.
   */
  modelbox::Status GetObject(const ObsOptions &opt,
                             const std::string &file_local_path);

  /**
   * @brief   Get buffer from an OBS object.
   * @param   opt - in, obs option.
   * @param   buf - out, buffer.
   * @param   size - in, get buffer size.
   * @param   offset - in, start byte to get.
   * @return  modelbox::STATUS_OK - Successfully get the buffer.
   *          modelbox::STATUS_AGAIN - Need to try again.
   *          other status - Failed.
   */
  modelbox::Status GetBuffer(ObsOptions &opt, unsigned char *buf, uint64_t size,
                             uint64_t offset);

  /**
   * @brief   Get object size from OBS.
   * @param   opt - in, obs option.
   * @return  Object size.
   */
  uint64_t GetObjectSize(ObsOptions &opt);

  /**
   * @brief   Put an object to OBS.
   * @param   opt - in, OBS options.
   * @param   data - in, the object would be downloaded to this path.
   * @param   data_size
   *                            Notes: 1. directories would be made recursively
   * if those do not exist.
   *                                   2. Read-Write right is needed to access
   * the path.
   * @return  modelbox::STATUS_OK - Successfully get the list.
   *          modelbox::STATUS_AGAIN - Need to try again.
   *          other status - Failed.
   */
  modelbox::Status PutObject(const ObsOptions &opt, const char *data,
                             size_t data_size);

  virtual ~ObsClient();

  static std::mutex obs_client_lock_;

 private:
  ObsClient();

  /**
   * @brief   Initialize the OBS SDK.
   * @return  Successful or not
   */
  modelbox::Status InitObsSdk();

  /**
   * @brief   Deinitialize the OBS SDK.
   * @return  void
   */
  void DeInitObsSdk();

  /**
   * @brief   get Ak/Sk/SecurityToken from hw_auth.
   * @param   domain_name - in, user domain name
   * @param   xrole_name - in, user xrole name to vas
   * @param   access_key - out, AK
   * @param   secret_key - out, SK
   * @param   security_token - out, Security Token
   * @return  Successful or not
   */
  modelbox::Status GetAuthInfo(const std::string &domain_name,
                               const std::string &xrole_name,
                               const std::string &user_id,
                               std::string &access_key, std::string &secret_key,
                               std::string &security_token);

  /**
   * @brief   Notify hw_auth to update the Ak/SK/SecurityToken, and get the
   * updated ones.
   * @param   domain_name - in, user domain name
   * @param   xrole_name - in, user xrole name to vas
   * @param   access_key - out, AK
   * @param   secret_key - out, SK
   * @param   security_token - out, Security Token
   * @return  Successful or not
   */
  modelbox::Status GetUpdatedAuthInfo(const std::string &domain_name,
                                      const std::string &xrole_name,
                                      const std::string &user_id,
                                      std::string &access_key,
                                      std::string &secret_key,
                                      std::string &security_token);

  /**
   * @brief Notify hw_auth to update the Ak/SK/SecurityToken
   * @param output_info - identify the configuration
   * @return Successful or not
   */
  modelbox::Status NotifyToUpdateAuthInfo(const std::string &domain_name,
                                          const std::string &xrole_name);

  /**
   * @brief   Validate the OBS options except for ObsOptions::path.
   * @param   opt - in, OBS options
   * @return  true - Valid, false - Invalid
   */
  bool IsValidOptionExceptPath(const ObsOptions &opt);

  /**
   * @brief   Validate the OBS options including ObsOptions::path.
   * @param   opt - in, OBS options
   * @return  true - Valid, false - Invalid
   */
  bool IsValidOptionIncludingPath(const ObsOptions &opt);

  /**
   * @brief
   * @param   src - in, ObsOptions
   * @param   dst - out, obs_options from OBS SDK
   * @return
   */
  void SetObsOption(const ObsOptions &src, const std::string &ak,
                    const std::string &sk, const std::string &security_token,
                    obs_options &dst);

  /**
   * @brief   Open the file to accept downloaded data. The file would be created
   * if not exists.
   * @param   full_file_path - in, Path to save the downloaded OBS file,
   * including the file name.
   * @return
   */
  std::shared_ptr<FILE> OpenLocalFile(const std::string &full_file_path);

  /**
   * @brief Based on the obs_status, notify the hw_auth to update the auth info.
   * @param status - status return from OBS SDK
   * @return need or not.
   */
  bool NeedUpdateAuthInfo(obs_status status);

  /**
   * @brief Based on the obs_status, notify the framework to try uploading data
   * again.
   * @param status - status return from OBS SDK
   * @return need or not.
   */
  bool NeedTryAgain(obs_status status);
};

}  // namespace modelbox

#endif  // MODELBOX_FLOWUNIT_OBS_CLIENT_H_