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

#include "js_engine.h"
#ifdef ENABLE_JS_PLUGIN
#include <fstream>
#include <iostream>

#include "modelbox/base/log.h"
#include "securec.h"

namespace modelbox {

void fatal_function(void *udata, const char *msg) {
  MBLOG_FATAL << "Duktape fatal error, detail:" << msg << std::endl;
  abort();  // If not abort, duktape engine will stuck
}

JSFunctionParam::JSFunctionParam(std::shared_ptr<duk_context> ctx)
    : ctx_(ctx) {}

void JSFunctionParam::AddBoolean(bool val) {
  ++argc_;
  duk_push_boolean(ctx_.get(), val);
}

void JSFunctionParam::AddInt32(int32_t val) {
  ++argc_;
  duk_push_int(ctx_.get(), val);
}

void JSFunctionParam::AddUint32(uint32_t val) {
  ++argc_;
  duk_push_uint(ctx_.get(), val);
}

void JSFunctionParam::AddNum(double val) {
  ++argc_;
  duk_push_number(ctx_.get(), val);
}

void *JSFunctionParam::AddBuffer(size_t size) {
  ++argc_;
  return duk_push_fixed_buffer(ctx_.get(), size);
}

void JSFunctionParam::AddString(const std::string &val) {
  ++argc_;
  duk_push_lstring(ctx_.get(), val.c_str(), val.size());
}

void JSFunctionParam::AddPointer(void *val) {
  ++argc_;
  duk_push_pointer(ctx_.get(), val);
}

void JSFunctionParam::AddNull() {
  ++argc_;
  duk_push_null(ctx_.get());
}

void JSFunctionParam::AddHeapPtr(void *ptr) {
  ++argc_;
  duk_push_heapptr(ctx_.get(), ptr);
}

size_t JSFunctionParam::GetParamSize() { return argc_; }

JSFunctionReturn::JSFunctionReturn(std::shared_ptr<duk_context> ctx)
    : ctx_(ctx) {}

bool JSFunctionReturn::GetBool() { return duk_get_boolean(ctx_.get(), -1); }

int32_t JSFunctionReturn::GetInt32() { return duk_get_int(ctx_.get(), -1); }

uint32_t JSFunctionReturn::GetUint32() { return duk_get_uint(ctx_.get(), -1); }

double JSFunctionReturn::GetNum() { return duk_get_number(ctx_.get(), -1); }

std::string JSFunctionReturn::GetString() {
  auto duk_str = duk_get_string(ctx_.get(), -1);
  if (duk_str == nullptr) {
    return "";
  }

  return duk_str;
}

void *JSFunctionReturn::GetPointer() { return duk_get_pointer(ctx_.get(), -1); }

JSCtx::JSCtx() = default;

modelbox::Status JSCtx::Init() {
  auto ctx_ptr =
      duk_create_heap(nullptr, nullptr, nullptr, nullptr, fatal_function);
  if (ctx_ptr == nullptr) {
    std::string err = "Init js ctx failed";
    MBLOG_ERROR << err;
    return modelbox::STATUS_FAULT;
  }

  ctx_.reset(ctx_ptr, [](duk_context *ctx) { duk_destroy_heap(ctx); });
  duk_push_global_object(ctx_.get());
  return modelbox::STATUS_OK;
}

modelbox::Status JSCtx::LoadCode(const std::string &code,
                                 const std::string &code_name) {
  auto handle = WaitCtx();
  auto buf = (char *)duk_push_fixed_buffer(ctx_.get(), code.size());
  if (buf == nullptr) {
    std::string err =
        "Create buffer in js ctx failed, size " + std::to_string(code.size());
    MBLOG_ERROR << err;
    return {STATUS_FAULT, err};
  }

  auto e_ret = memcpy_s(buf, code.size(), code.data(), code.size());
  if (e_ret != EOK) {
    std::string err = "memcpy failed, size " + std::to_string(code.size()) +
                      ", ret " + std::to_string(e_ret);
    MBLOG_ERROR << err;
    return {STATUS_FAULT, err};
  }

  duk_buffer_to_string(ctx_.get(), -1);
  duk_push_string(ctx_.get(), code_name.c_str());
  auto ret = CompileCode();
  if (!ret) {
    return ret;
  }

  MBLOG_INFO << "Load source code success";
  return modelbox::STATUS_OK;
}

modelbox::Status JSCtx::LoadSource(const std::string &source_path) {
  std::ifstream file(source_path);
  if (!file.is_open()) {
    std::string err = "Open file " + source_path + " failed";
    MBLOG_ERROR << err;
    return {STATUS_FAULT, err};
  }

  file.seekg(0, std::ifstream::end);
  size_t file_len = file.tellg();
  const size_t max_file_size = 1 * 1024 * 1024;
  if (file_len > max_file_size) {
    std::string err = "Size of file " + source_path + " is great than 1MB";
    MBLOG_ERROR << err;
    return {STATUS_FAULT, err};
  }

  auto handle = WaitCtx();
  auto buf = (char *)duk_push_fixed_buffer(ctx_.get(), file_len);
  if (buf == nullptr) {
    std::string err =
        "Create buffer in js ctx failed, size " + std::to_string(file_len);
    MBLOG_ERROR << err;
    return {STATUS_FAULT, err};
  }

  file.seekg(0, std::ifstream::beg);
  file.read(buf, file_len);
  file.close();

  duk_buffer_to_string(ctx_.get(), -1);
  duk_push_string(ctx_.get(), source_path.c_str());
  auto ret = CompileCode();
  if (!ret) {
    return ret;
  }

  MBLOG_INFO << "Load source " << source_path << " success";
  return STATUS_OK;
}

modelbox::Status JSCtx::CompileCode() {
  auto ret = duk_pcompile(ctx_.get(), DUK_COMPILE_EVAL);
  if (ret != 0) {
    std::string err = "Load source failed";
    auto duk_errmsg = duk_safe_to_string(ctx_.get(), -1);
    if (duk_errmsg) {
      err += ", err:";
      err += duk_errmsg;
    }
    MBLOG_ERROR << err;
    duk_pop(ctx_.get());  // duk_pcomile
    return {STATUS_FAULT, err};
  }

  duk_push_global_object(ctx_.get());
  ret = duk_pcall_method(ctx_.get(), 0);
  if (ret != 0) {
    std::string err = "Load source failed, err:";
    auto duk_errmsg = duk_safe_to_string(ctx_.get(), -1);
    if (duk_errmsg) {
      err += duk_errmsg;
    }

    MBLOG_ERROR << err;
    duk_pop(ctx_.get());  // pop duk_pcall_method return
    return {STATUS_FAULT, err};
  }

  duk_pop(ctx_.get());  // pop duk_pcall_method return
  return modelbox::STATUS_OK;
}

modelbox::Status JSCtx::RegisterFunc(const std::string &name,
                                     duk_c_function func, int32_t argc) {
  auto handle = WaitCtx();
  duk_push_c_function(ctx_.get(), func, argc);
  duk_put_global_string(ctx_.get(), name.c_str());
  return STATUS_OK;
}

modelbox::Status JSCtx::CallFunc(const std::string &func_name,
                                 const FillParam &fill_param,
                                 const ReadReturn &read_return) {
  auto handle = WaitCtx();
  auto b_ret = duk_get_prop_string(ctx_.get(), -1, func_name.c_str());
  if (!b_ret) {
    duk_pop(ctx_.get());  // duk_get_prop_string
    std::string err = "func " + func_name + " not defined";
    MBLOG_ERROR << err;
    return {STATUS_FAULT, err};
  }

  JSFunctionParam func_param(ctx_);
  fill_param(func_param);

  auto ret = duk_pcall(ctx_.get(), func_param.GetParamSize());
  if (ret != 0) {
    auto err = duk_safe_to_string(ctx_.get(), -1);
    std::string err_msg = "call " + func_name + " failed";
    if (err) {
      err_msg += ", err:";
      err_msg += err;
    }
    duk_pop(ctx_.get());  // duk_pcall
    MBLOG_ERROR << err_msg;
    return {STATUS_FAULT, err_msg};
  }

  JSFunctionReturn func_ret(ctx_);
  read_return(func_ret);
  duk_pop(ctx_.get());  // duk_pcall
  return STATUS_OK;
}

}  // namespace modelbox

#endif  // ENABLE_JS_PLUGIN