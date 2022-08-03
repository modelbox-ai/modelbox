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

#include <map>
#include <mutex>

#include "server_plugin.h"

#ifdef ENABLE_JS_PLUGIN
#include "js_engine.h"
#endif

namespace modelbox {

std::mutex JsPlugin::runtime_to_plugin_lock;
std::map<void *, JsPlugin *> JsPlugin::runtime_to_plugin;

#ifdef ENABLE_JS_PLUGIN

std::set<modelbox::StatisticsNotifyType> NotifyTypesFromUint(
    uint32_t type_mask) {
  std::set<modelbox::StatisticsNotifyType> type_list;
  if (type_mask & (uint32_t)modelbox::StatisticsNotifyType::CREATE) {
    type_list.insert(modelbox::StatisticsNotifyType::CREATE);
  }

  if (type_mask & (uint32_t)modelbox::StatisticsNotifyType::CHANGE) {
    type_list.insert(modelbox::StatisticsNotifyType::CHANGE);
  }

  if (type_mask & (uint32_t)modelbox::StatisticsNotifyType::DELETE) {
    type_list.insert(modelbox::StatisticsNotifyType::DELETE);
  }

  if (type_mask & (uint32_t)modelbox::StatisticsNotifyType::TIMER) {
    type_list.insert(modelbox::StatisticsNotifyType::TIMER);
  }

  return type_list;
}

static duk_ret_t JSRegisterStatsNotify(duk_context *ctx) {
  const auto *path_pattern = duk_require_string(ctx, 0);
  auto type_mask = duk_require_uint(ctx, 1);
  const auto *func_name = duk_require_string(ctx, 2);
  auto *priv_data = duk_get_heapptr(ctx, 3);

  if (func_name == nullptr || path_pattern == nullptr) {
    MBLOG_ERROR << "param is invalid.";
    return -1;
  }

  MBLOG_INFO << "JSPlugin register notify[path:" << path_pattern
             << ",type:" << type_mask << ",func:" << func_name
             << ",priv_data:" << priv_data << "]";

  auto *plugin_ctx = JsPlugin::GetPlugin(ctx);
  if (plugin_ctx == nullptr) {
    MBLOG_ERROR << "plugin_ctx is null";
    return -1;  // return error to js
  }

  auto type_list = NotifyTypesFromUint(type_mask);
  if (type_list.empty()) {
    MBLOG_ERROR << "register type should not be empty";
    return 0;
  }

  ((JsPlugin *)plugin_ctx)
      ->RegisterStatsNotify(path_pattern, type_list, func_name, priv_data);
  return 0;  // means this function has no return value in value stack
}

/**
 * @note Not recommended
 */
static duk_ret_t JSRegisterStatsTimerNotify(duk_context *ctx) {
  const auto *path_pattern = duk_require_string(ctx, 0);
  auto timer_delay = duk_require_uint(ctx, 1);
  auto timer_interval = duk_require_uint(ctx, 2);
  const auto *func_name = duk_require_string(ctx, 3);
  auto *priv_data = duk_get_heapptr(ctx, 4);
  if (path_pattern == nullptr || func_name == nullptr) {
    MBLOG_ERROR << "register param is invalid";
    return -1;
  }
  MBLOG_INFO << "JSPlugin register timer notify[path:" << path_pattern
             << ",delay:" << timer_delay << ",interval:" << timer_interval
             << ",func:" << func_name << ",priv_data:" << priv_data << "]";

  auto *plugin_ctx = JsPlugin::GetPlugin(ctx);
  if (plugin_ctx == nullptr) {
    MBLOG_ERROR << "plugin_ctx is null";
    return -1;  // return error to js
  }

  ((JsPlugin *)plugin_ctx)
      ->RegisterStatsNotify(path_pattern,
                            {modelbox::StatisticsNotifyType::TIMER}, func_name,
                            priv_data, timer_delay, timer_interval);
  return 0;
}

static duk_ret_t JSGetStatsValue(duk_context *ctx) {
  const auto *path = duk_require_string(ctx, 0);
  auto stats = modelbox::Statistics::GetGlobalItem();
  if (stats == nullptr || path == nullptr) {
    MBLOG_ERROR << "Global item is invalid.";
    return -1;
  }

  auto item = stats->GetItem(path);
  if (item == nullptr) {
    MBLOG_ERROR << "Get value for " << path << " failed";
    duk_push_null(ctx);
  } else {
    duk_push_string(ctx, item->GetValue()->ToString().c_str());
  }

  return 1;  // means this function has return value in value stack
}

static duk_ret_t JSRouteData(duk_context *ctx) {
  const auto *topic = duk_require_string(ctx, 0);
  const auto *msg_name = duk_require_string(ctx, 1);
  if (topic == nullptr || msg_name == nullptr) {
    MBLOG_ERROR << "get message name failed.";
    return -1;
  }

  auto msg_data = std::make_shared<std::string>();
  *msg_data = duk_require_string(ctx, 2);
  if (msg_data == nullptr) {
    MBLOG_ERROR << "get message data failed.";
    return -1;
  }

  MBLOG_DEBUG << "Send data to " << topic << ", " << msg_name << ", "
              << *msg_data;
  std::shared_ptr<const void> buffer;
  auto router = PluginMsgRouter::GetInstance();
  buffer.reset(msg_data->data(), [msg_data](const void *ptr) {});
  auto ret = router->RouteMsg(topic, msg_name, buffer, msg_data->size());
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "JSPlugin: route msg failed: " << topic << ", " << msg_name
                << ", " << *msg_data;
  }

  return 0;
}

static void GetCallerInfo(duk_context *ctx, std::string &file_name,
                          int32_t &line_no) {
  line_no = 0;
  file_name = "unknown";
  duk_inspect_callstack_entry(ctx, -3);
  if (duk_is_object(ctx, -1)) {
    duk_get_prop_string(ctx, -1, "lineNumber");
    line_no = duk_to_int(ctx, -1);
    duk_pop(ctx);

    duk_get_prop_string(ctx, -1, "function");

    duk_get_prop_string(ctx, -1, "fileName");
    const auto *js_file_name = duk_to_string(ctx, -1);
    if (js_file_name) {
      file_name = js_file_name;
    }
    duk_pop(ctx);

    duk_pop(ctx);
  }

  duk_pop(ctx);  // Callstack
}

static duk_ret_t JSModelboxLog(duk_context *ctx) {
  const auto *level = duk_require_string(ctx, 0);
  const auto *msg = duk_require_string(ctx, 1);

  if (level == nullptr || msg == nullptr) {
    MBLOG_ERROR << "input param is invalid";
    return -1;
  }

  std::string file_name;
  int32_t line_no;
  GetCallerInfo(ctx, file_name, line_no);

  std::string pre_fix =
      "[ " + file_name + ":" + std::to_string(line_no) + " ] ";
  auto print_msg = pre_fix + msg;
  if (level == std::string("fatal")) {
    MBLOG_FATAL << print_msg;
  } else if (level == std::string("error")) {
    MBLOG_ERROR << print_msg;
  } else if (level == std::string("warn")) {
    MBLOG_WARN << print_msg;
  } else if (level == std::string("notice")) {
    MBLOG_NOTICE << print_msg;
  } else if (level == std::string("info")) {
    MBLOG_INFO << print_msg;
  } else {
    MBLOG_DEBUG << print_msg;
  }

  return 0;
}

JsPlugin::JsPlugin(const std::string &plugin_path)
    : ServerPlugin(plugin_path), js_ctx_(std::make_shared<JSCtx>()) {}

JsPlugin::~JsPlugin() { DelMap(js_ctx_->GetRuntime()); }

modelbox::Status JsPlugin::Init(
    std::shared_ptr<modelbox::Configuration> config) {
  auto ret = js_ctx_->Init();
  if (!ret) {
    return ret;
  }

  AddMap(js_ctx_->GetRuntime(), this);
  (void)RegisterCFunction();
  ret = LoadInitCode();
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "Load init code failed";
    return ret;
  }

  ret = js_ctx_->LoadSource(plugin_path_);
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "Load plugin " << plugin_path_ << " failed";
    return ret;
  }

  int32_t js_func_ret = -1;
  ret = js_ctx_->CallFunc(
      "init", [this](JSFunctionParam &param) { param.AddPointer(this); },
      [&js_func_ret](JSFunctionReturn &ret) { js_func_ret = ret.GetInt32(); });
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "Call plugin init " << plugin_path_ << " failed";
    return modelbox::STATUS_FAULT;
  }

  if (js_func_ret != 0) {
    MBLOG_ERROR << "Plugin init " << plugin_path_
                << " failed, ret:" << js_func_ret;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status JsPlugin::RegisterCFunction() {
  js_ctx_->RegisterFunc("registerStatsNotify", JSRegisterStatsNotify, 4);
  js_ctx_->RegisterFunc("registerStatsTimerNotify", JSRegisterStatsTimerNotify,
                        5);
  js_ctx_->RegisterFunc("getStatsValue", JSGetStatsValue, 1);
  js_ctx_->RegisterFunc("routeData", JSRouteData, 3);
  js_ctx_->RegisterFunc("modelboxLog", JSModelboxLog, 2);
  return modelbox::STATUS_OK;
}

modelbox::Status JsPlugin::LoadInitCode() {
  std::string init_var_code = R"(
    var NOTIFY_CREATE = 1;
    var NOTIFY_DELETE = 2;
    var NOTIFY_CHANGE = 4;
    var NOTIFY_TIMER = 8;

    var console = {};

    console.log = function (msg) {
      modelboxLog("info", msg);
    }

    console.info = console.log

    console.warn = function (msg) {
      modelboxLog("warn", msg);
    }

    console.error = function (msg) {
      modelboxLog("error", msg);
    }
  )";
  return js_ctx_->LoadCode(init_var_code, "JsPluginInitCode");
}

modelbox::Status JsPlugin::Start() {
  int32_t js_func_ret = -1;
  auto ret = js_ctx_->CallFunc(
      "start", [this](JSFunctionParam &param) { param.AddPointer(this); },
      [&js_func_ret](JSFunctionReturn &ret) { js_func_ret = ret.GetInt32(); });
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "Call plugin start " << plugin_path_ << " failed";
    return modelbox::STATUS_FAULT;
  }

  if (js_func_ret != 0) {
    MBLOG_ERROR << "Plugin start " << plugin_path_
                << " failed, ret:" << js_func_ret;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status JsPlugin::Stop() {
  int32_t js_func_ret = -1;
  auto stats = modelbox::Statistics::GetGlobalItem();
  for (auto &notify_cfg : notify_cfg_list_) {
    stats->UnRegisterNotify(notify_cfg);
  }

  auto ret = js_ctx_->CallFunc(
      "stop", [this](JSFunctionParam &param) { param.AddPointer(this); },
      [&js_func_ret](JSFunctionReturn &ret) { js_func_ret = ret.GetInt32(); });
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "Call plugin stop " << plugin_path_ << " failed";
    return modelbox::STATUS_FAULT;
  }

  if (js_func_ret != 0) {
    MBLOG_ERROR << "Plugin stop " << plugin_path_
                << " failed, ret:" << js_func_ret;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

void JsPlugin::RegisterStatsNotify(
    const std::string &path_pattern,
    const std::set<modelbox::StatisticsNotifyType> &type_list,
    const std::string &func_name, void *priv_data, size_t delay,
    size_t interval) {
  auto stats = modelbox::Statistics::GetGlobalItem();
  std::weak_ptr<JSCtx> js_ctx_ref = js_ctx_;
  auto notify_cfg = std::make_shared<modelbox::StatisticsNotifyCfg>(
      path_pattern,
      [js_ctx_ref, func_name, priv_data](
          const std::shared_ptr<const modelbox::StatisticsNotifyMsg> &msg) {
        auto js_ctx_ptr = js_ctx_ref.lock();
        if (js_ctx_ptr == nullptr) {
          return;
        }

        js_ctx_ptr->CallFunc(func_name, [&](JSFunctionParam &param) {
          param.AddString(msg->path_);
          std::string value_str;
          if (msg->value_ != nullptr) {
            value_str = msg->value_->ToString();
          }
          param.AddString(value_str);
          param.AddUint32((uint32_t)msg->type_);
          param.AddHeapPtr(priv_data);
        });
      },
      type_list);
  if (type_list.find(modelbox::StatisticsNotifyType::TIMER) !=
      type_list.end()) {
    notify_cfg->SetNotifyTimer(delay, interval);
  }

  notify_cfg_list_.push_back(notify_cfg);
  stats->RegisterNotify(notify_cfg);
}
#else   // ENABLE_JS_PLUGIN
JsPlugin::JsPlugin(const std::string &plugin_path)
    : ServerPlugin(plugin_path) {}
JsPlugin::~JsPlugin() = default;

modelbox::Status JsPlugin::Init(
    std::shared_ptr<modelbox::Configuration> config) {
  MBLOG_ERROR << "Js plugin is not enabled, please remove [" << plugin_path_
              << "] from conf";
  return modelbox::STATUS_NOTSUPPORT;
}
modelbox::Status JsPlugin::Start() { return modelbox::STATUS_NOTSUPPORT; }
modelbox::Status JsPlugin::Stop() { return modelbox::STATUS_NOTSUPPORT; }

void JsPlugin::RegisterStatsNotify(
    const std::string &path_pattern,
    const std::set<modelbox::StatisticsNotifyType> &type_list,
    const std::string &func_name, void *priv_data, size_t delay,
    size_t interval) {}
#endif  // ENABLE_JS_PLUGIN

void JsPlugin::AddMap(void *runtime, JsPlugin *plugin) {
  std::lock_guard<std::mutex> lck(runtime_to_plugin_lock);
  runtime_to_plugin[runtime] = plugin;
}

void JsPlugin::DelMap(void *runtime) {
  std::lock_guard<std::mutex> lck(runtime_to_plugin_lock);
  runtime_to_plugin.erase(runtime);
}

JsPlugin *JsPlugin::GetPlugin(void *runtime) {
  std::lock_guard<std::mutex> lck(runtime_to_plugin_lock);
  auto item = runtime_to_plugin.find(runtime);
  if (item == runtime_to_plugin.end()) {
    return nullptr;
  }

  return item->second;
}

}  // namespace modelbox