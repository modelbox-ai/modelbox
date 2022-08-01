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

#include "model_decrypt.h"

#include <dirent.h>
#include <dlfcn.h>
#include <limits.h>
#include <sys/stat.h>

#include <cstdint>
#include <fstream>

#include "model_decrypt_header.h"
#include "modelbox/base/log.h"
#include "modelbox/base/status.h"
#include "modelbox/base/utils.h"

ModelDecryption::~ModelDecryption() {
  if (fmodel_.is_open()) {
    fmodel_.close();
  }
}

modelbox::Status ModelDecryption::Init(
    const std::string& model_path,
    const std::shared_ptr<modelbox::Drivers>& drivers_ptr,
    const std::shared_ptr<modelbox::Configuration>& config) {
  model_state_ = MODEL_STATE_ERROR;
  header_offset_ = 0;

  Defer {
    if (model_state_ == MODEL_STATE_ERROR && fmodel_.is_open()) {
      fmodel_.close();
    }
  };

  fmodel_.open(model_path, std::ios::binary);
  if (fmodel_.fail() || !fmodel_.is_open()) {
    std::string errmsg = "open model '";
    errmsg += model_path + "' failed, " + modelbox::StrError(errno);
    MBLOG_ERROR << errmsg;
    return {modelbox::STATUS_INVALID, errmsg};
  }

  fmodel_.seekg(0, std::ios::end);
  fsize_ = fmodel_.tellg();
  if (fsize_ <= 0) {
    std::string errmsg = "empty model file: " + model_path;
    MBLOG_ERROR << errmsg;
    return {modelbox::STATUS_BADCONF, errmsg};
  }

  auto plugin_name = config->GetString("encryption.plugin_name");
  auto plugin_version = config->GetString("encryption.plugin_version");
  if (plugin_name.empty()) {
    GetInfoFromHeader(plugin_name, plugin_version, config);
  }

  if (!plugin_name.empty()) {
    if (drivers_ptr == nullptr) {
      MBLOG_ERROR << "drivers_ptr is null";
      return modelbox::STATUS_FAULT;
    }

    auto plugin_driver = drivers_ptr->GetDriver(
        DRIVER_CLASS_MODEL_DECRYPT, DRIVER_TYPE, plugin_name, plugin_version);
    if (plugin_driver == nullptr) {
      std::string errmsg = "Can not find decrytp drivers: " + plugin_name;
      MBLOG_ERROR << errmsg;
      // fclose will call when ~ModelDecryption
      return {modelbox::STATUS_BADCONF, errmsg};
    }

    cur_factory_ = plugin_driver->CreateFactory();
    if (cur_factory_ == nullptr) {
      MBLOG_ERROR << "Plugin : " << plugin_name << " factory create failed";
      return {modelbox::STATUS_FAULT, "decrypt driver create failed"};
    }

    cur_plugin_ = std::dynamic_pointer_cast<IModelDecryptPlugin>(
        cur_factory_->GetDriver());
    if (cur_plugin_ == nullptr) {
      MBLOG_ERROR << "plugin : " << plugin_name
                  << " is not derived from IModelDecryptPlugin";
      return {modelbox::STATUS_FAULT, "decrypt driver create failed"};
    }

    auto ret = cur_plugin_->Init(model_path, config);
    if (ret == modelbox::STATUS_SUCCESS) {
      model_state_ = MODEL_STATE_ENCRYPT;
    } else {
      MBLOG_ERROR << "drivers Init Error";
      return ret;
    }
  } else {
    model_state_ = MODEL_STATE_PLAIN;
  }

  return modelbox::STATUS_SUCCESS;
}

void ModelDecryption::GetInfoFromHeader(
    std::string& plugin_name, std::string& plugin_version,
    const std::shared_ptr<modelbox::Configuration>& config) {
  struct PrefixInfo model_info;
  fmodel_.seekg(0, std::ios::beg);
  fmodel_.read(reinterpret_cast<char*>(&model_info), sizeof(PrefixInfo));

  std::string magic_str(model_info.magic, MAGIC_SIZE);
  if (fmodel_.gcount() != sizeof(PrefixInfo) || magic_str != MAGIC_FLAG) {
    // here is plain model , not err , so do not log
    plugin_name = "";
    plugin_version = "";
    return;
  }

  plugin_name = std::string(model_info.plugin_name);
  plugin_version = std::to_string(model_info.ver_major) + "." +
                   std::to_string(model_info.ver_minor) + "." +
                   std::to_string(model_info.ver_patch);
  config->SetProperty("encryption.header_reserve", (uint8_t)model_info.reserve);
  header_offset_ = (int32_t)(sizeof(PrefixInfo));
}

uint8_t* ModelDecryption::GetModelBuffer(int64_t& model_len) {
  model_len = 0;
  if (model_state_ == MODEL_STATE_ERROR) {
    MBLOG_ERROR << "model_state is error";
    modelbox::StatusError = {modelbox::STATUS_INVALID, "model_state is error"};
    return nullptr;
  }

  // tensorflow TF_Buffer seems a c-style code, here use std::malloc
  auto* model_buf = static_cast<uint8_t*>(malloc(fsize_));
  if (!model_buf) {
    MBLOG_ERROR << "memory alloc fail with size =." << fsize_;
    modelbox::StatusError = {modelbox::STATUS_NOMEM, "Read file fail."};
    return nullptr;
  }

  fmodel_.seekg(0, std::ios::beg);
  fmodel_.read((char*)model_buf, fsize_);
  if (fmodel_.gcount() != fsize_) {
    MBLOG_ERROR << "Read file fail.";
    free(model_buf);
    modelbox::StatusError = {modelbox::STATUS_INVALID, "Read file fail."};
    return nullptr;
  }

  if (model_state_ == MODEL_STATE_ENCRYPT && cur_plugin_ != nullptr) {
    int64_t raw_len = fsize_ - header_offset_;
    int64_t plain_len = raw_len + EVP_MAX_BLOCK_LENGTH + 1;
    auto* plain_buf = static_cast<uint8_t*>(malloc(raw_len));
    auto ret = cur_plugin_->ModelDecrypt(model_buf + header_offset_, raw_len,
                                         plain_buf, plain_len);
    free(model_buf);

    if (ret != modelbox::STATUS_SUCCESS) {
      MBLOG_ERROR << "ModelDecrypt fail.";
      model_len = 0;
      free(plain_buf);
      modelbox::StatusError = {ret, "ModelDecrypt fail."};
      return nullptr;
    }
    model_len = plain_len;
    return plain_buf;
  }
  model_len = fsize_;
  return model_buf;
}

std::shared_ptr<uint8_t> ModelDecryption::GetModelSharedBuffer(
    int64_t& model_len) {
  uint8_t* ret_buf = GetModelBuffer(model_len);
  if (ret_buf) {
    std::shared_ptr<uint8_t> retShare(ret_buf, [](uint8_t* p) { free(p); });
    return retShare;
  }
  return nullptr;
}
