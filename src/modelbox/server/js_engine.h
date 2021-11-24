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

#ifndef MODELBOX_JAVASCRIPT_ENGINE_H_
#define MODELBOX_JAVASCRIPT_ENGINE_H_

#ifdef ENABLE_JS_PLUGIN
#include <duktape.h>
#include <modelbox/base/status.h>
#include <modelbox/base/utils.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace modelbox {

class JSFunctionParam {
 public:
  JSFunctionParam(std::shared_ptr<duk_context> ctx);
  virtual ~JSFunctionParam() = default;

  void AddBoolean(bool val);
  void AddInt32(int32_t val);
  void AddUint32(uint32_t val);
  void AddNum(double val);
  void *AddBuffer(size_t size);
  void AddString(const std::string &val);
  void AddPointer(void *val);
  void AddNull();
  void AddHeapPtr(void *ptr);

  size_t GetParamSize();

 private:
  std::shared_ptr<duk_context> ctx_;
  size_t argc_{0};
};

class JSFunctionReturn {
 public:
  JSFunctionReturn(std::shared_ptr<duk_context> ctx);
  virtual ~JSFunctionReturn() = default;

  bool GetBool();
  int32_t GetInt32();
  uint32_t GetUint32();
  double GetNum();
  std::string GetString();
  void *GetPointer();

 private:
  std::shared_ptr<duk_context> ctx_;
};

using FillParam = std::function<void(JSFunctionParam &param)>;
using ReadReturn = std::function<void(JSFunctionReturn &ret)>;

class JSCtx {
 public:
  JSCtx();
  virtual ~JSCtx() = default;

  /**
   * @brief Init js context
   * @return Result of init
   */
  modelbox::Status Init();

  /**
   * @brief Load js source code to ctx
   * @param code Js source code
   * @param code_name To indentify the source code
   * @return Result of load
   */
  modelbox::Status LoadCode(const std::string &code,
                            const std::string &code_name);

  /**
   * @brief Load js source file to ctx
   * @param source_path Indicate js source file to load
   * @return Result of load
   */
  modelbox::Status LoadSource(const std::string &source_path);

  /**
   * @brief Register target c function to js ctx
   * @param name Function name which is visiable in js
   * @param func Target function
   * @param argc Arg number of target function
   * @return Result of register
   */
  modelbox::Status RegisterFunc(const std::string &name, duk_c_function func,
                                int32_t argc);

  /**
   * @brief Call target function in js
   * @param func_name Target function name
   * @param fill_param Func to fill param for target function
   * @param read_return Func to read target function result
   * @return Result of call func operation
   */
  modelbox::Status CallFunc(const std::string &func_name,
                            const FillParam &fill_param = FillParamDefault,
                            const ReadReturn &read_return = ReadReturnDefault);

  /**
   * @brief Get runtime pointer
   * @return Runtime pointer
   */
  void *GetRuntime() const { return ctx_.get(); }

  /**
   * @brief Defaul fill param func, if target function needs no param
   * @param param Param for target function
   */
  static void FillParamDefault(JSFunctionParam &param){};

  /**
   * @brief Defaul read return func, if you do not care target function return
   * @param param Return of target function
   */
  static void ReadReturnDefault(JSFunctionReturn &ret){};

 private:
  modelbox::Status CompileCode();

  inline std::shared_ptr<modelbox::DeferGuard> WaitCtx() {
    // need this to ensure excution order
    // avoid starving caused by direct lock
    auto my_seq = waiting_seq_.fetch_add(1);
    std::unique_lock<std::mutex> lck(ctx_running_lock_);
    ctx_running_cv_.wait(lck,
                         [this, my_seq]() { return my_seq == execute_seq_; });
    auto defer_guard = std::make_shared<modelbox::DeferGuard>([this]() {
      std::unique_lock<std::mutex> lck(ctx_running_lock_);
      ++execute_seq_;
      ctx_running_cv_.notify_all();
    });

    return defer_guard;
  }

  std::shared_ptr<duk_context> ctx_;
  std::atomic_uint64_t waiting_seq_{0};
  std::atomic_uint64_t execute_seq_{0};
  std::mutex ctx_running_lock_;
  std::condition_variable ctx_running_cv_;
};

}  // namespace modelbox

#endif  // ENABLE_JS_PLUGIN
#endif  // MODELBOX_JAVASCRIPT_ENGINE_H_