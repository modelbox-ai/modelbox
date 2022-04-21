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

#ifndef MODELBOX_FLOW_UNIT_H_
#define MODELBOX_FLOW_UNIT_H_

#include <modelbox/base/device.h>
#include <modelbox/base/driver.h>
#include <modelbox/base/log.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/buffer_list.h>
#include <modelbox/data_context.h>
#include <modelbox/stream.h>
#include <modelbox/tensor_list.h>
#include <string.h>

#include <functional>
#include <map>
#include <queue>
#include <regex>
#include <set>
#include <string>
#include <vector>

namespace modelbox {

constexpr const char *EVENT_PORT_NAME = "Event_Port";
constexpr const char *EXTERNAL_PORT_NAME = "External_Port";
constexpr const char *DRIVER_CLASS_FLOWUNIT = "DRIVER-FLOWUNIT";
constexpr uint32_t STREAM_DEFAULT_BATCH_SIZE = 1;
constexpr uint32_t NORMAL_DEFAULT_BATCH_SIZE = 8;
constexpr uint32_t STREAM_MAX_BATCH_SIZE = 1;
constexpr uint32_t NORMAL_MAX_BATCH_SIZE = 0;

using BufferPtr = std::shared_ptr<Buffer>;
using BufferPtrList = std::vector<BufferPtr>;

class SchedulerEvent;

enum FlowOutputType {
  ORIGIN = 0,
  EXPAND = 1,
  COLLAPSE = 2,
};

enum FlowType {
  STREAM = 0,
  NORMAL = 1,
};

enum ConditionType {
  NONE = 0,
  IF_ELSE = 1,
};

enum LoopType {
  NOT_LOOP = 0,
  LOOP = 1,
};

class FlowUnitPort {
 public:
  FlowUnitPort(const std::string &name) : port_name_(name){};
  FlowUnitPort(const std::string &name, const std::string &device_type)
      : port_name_(name), device_type_(device_type){};
  FlowUnitPort(const std::string &name, uint32_t device_mem_flags)
      : port_name_(name), device_mem_flags_(device_mem_flags){};
  FlowUnitPort(const std::string &name, const std::string &device_type,
               uint32_t device_mem_flags)
      : port_name_(name),
        device_type_(device_type),
        device_mem_flags_(device_mem_flags){};
  FlowUnitPort(const std::string &name, const std::string &device_type,
               const std::string &type)
      : port_name_(name), device_type_(device_type), port_type_(type){};
  FlowUnitPort(const std::string &name, const std::string &device_type,
               const std::string &type,
               const std::map<std::string, std::string> &ext)
      : port_name_(name),
        device_type_(device_type),
        port_type_(type),
        ext_(ext){};

  virtual ~FlowUnitPort(){};

  void SetDeviceType(const std::string &device_type) {
    device_type_ = device_type;
  };

  void SetPortName(const std::string &port_name) { port_name_ = port_name; };

  void SetPortType(const std::string &port_type) { port_type_ = port_type; };

  void SetDevice(std::shared_ptr<Device> device) { device_ = device; }

  void SetProperity(const std::string &key, const std::string &value) {
    ext_[key] = value;
  }

  std::string GetDeviceType() const { return device_type_; };
  std::string GetPortName() const { return port_name_; };
  std::string GetPortType() const { return port_type_; };
  std::shared_ptr<Device> GetDevice() const { return device_; }
  uint32_t GetDeviceMemFlags() const { return device_mem_flags_; }
  std::string GetProperity(const std::string &key) {
    if (ext_.find(key) == ext_.end()) {
      return "";
    }

    return ext_[key];
  }

 private:
  std::string port_name_;
  std::string device_type_;
  std::string port_type_;
  std::map<std::string, std::string> ext_;
  std::shared_ptr<Device> device_;
  uint32_t device_mem_flags_{0};
};

class FlowUnitInput : public FlowUnitPort {
 public:
  FlowUnitInput(const std::string &name) : FlowUnitPort(name){};
  FlowUnitInput(const std::string &name, const std::string &device_type)
      : FlowUnitPort(name, device_type){};
  FlowUnitInput(const std::string &name, uint32_t device_mem_flags)
      : FlowUnitPort(name, device_mem_flags){};
  FlowUnitInput(const std::string &name, const std::string &device_type,
                uint32_t device_mem_flags)
      : FlowUnitPort(name, device_type, device_mem_flags){};
  FlowUnitInput(const std::string &name, const std::string &device_type,
                const std::string &type)
      : FlowUnitPort(name, device_type, type){};
  FlowUnitInput(const std::string &name, const std::string &device_type,
                const std::string &type,
                const std::map<std::string, std::string> &ext)
      : FlowUnitPort(name, device_type, type, ext){};
  virtual ~FlowUnitInput(){};
};

class FlowUnitOutput : public FlowUnitPort {
 public:
  FlowUnitOutput(const std::string &name) : FlowUnitPort(name){};
  FlowUnitOutput(const std::string &name, uint32_t device_mem_flags)
      : FlowUnitPort(name, device_mem_flags){};
  /**
   * @deprecated
   **/
  FlowUnitOutput(const std::string &name, const std::string &device_type)
      : FlowUnitPort(name, device_type){};
  /**
   * @deprecated
   **/
  FlowUnitOutput(const std::string &name, const std::string &device_type,
                 uint32_t device_mem_flags)
      : FlowUnitPort(name, device_type, device_mem_flags){};
  /**
   * @deprecated
   **/
  FlowUnitOutput(const std::string &name, const std::string &device_type,
                 const std::string &type)
      : FlowUnitPort(name, device_type, type){};
  /**
   * @deprecated
   **/
  FlowUnitOutput(const std::string &name, const std::string &device_type,
                 const std::string &type,
                 const std::map<std::string, std::string> &ext)
      : FlowUnitPort(name, device_type, type, ext){};
  virtual ~FlowUnitOutput(){};
};

class FlowUnitOption {
 public:
  FlowUnitOption(const std::string &name, const std::string &type)
      : option_name_(name), option_type_(type), option_require_{false} {};
  FlowUnitOption(const std::string &name, const std::string &type, bool require)
      : option_name_(name), option_type_(type), option_require_{require} {};
  FlowUnitOption(const std::string &name, const std::string &type, bool require,
                 const std::string &default_value, const std::string &desc,
                 const std::map<std::string, std::string> &values)
      : option_name_(name),
        option_type_(type),
        option_require_(require),
        option_default_(default_value),
        option_desc_(desc),
        option_values_(values){};
  FlowUnitOption(const std::string &name, const std::string &type, bool require,
                 const std::string &default_value, const std::string &desc)
      : option_name_(name),
        option_type_(type),
        option_require_(require),
        option_default_(default_value),
        option_desc_(desc){};

  virtual ~FlowUnitOption() { option_values_.clear(); }

  void SetOptionName(const std::string &option_name) {
    option_name_ = option_name;
  }

  void SetOptionType(const std::string &option_type) {
    option_type_ = option_type;
  }

  void SetOptionRequire(bool option_require) {
    option_require_ = option_require;
  }

  void SetOptionDesc(const std::string &option_desc) {
    option_desc_ = option_desc;
  }

  void AddOptionValue(const std::string &key, const std::string &value) {
    option_values_.emplace(key, value);
  }

  std::string GetOptionName() const { return option_name_; }
  std::string GetOptionType() const { return option_type_; }
  bool IsRequire() const { return option_require_; }
  std::string GetOptionDefault() const { return option_default_; }
  std::string GetOptionDesc() const { return option_desc_; }
  std::map<std::string, std::string> GetOptionValues() {
    return option_values_;
  }
  std::string GetOptionValue(const std::string &key) {
    auto iter = option_values_.find(key);
    if (iter == option_values_.end()) {
      return "";
    }

    return option_values_[key];
  }

 private:
  std::string option_name_;
  std::string option_type_;
  bool option_require_{false};
  std::string option_default_;
  std::string option_desc_;
  std::map<std::string, std::string> option_values_;
};

class FlowUnitDesc {
 public:
  FlowUnitDesc()
      : output_type_(ORIGIN),
        flow_type_(NORMAL),
        condition_type_(NONE),
        loop_type_(NOT_LOOP),
        is_stream_same_count_(false),
        is_collapse_all_(false),
        is_exception_visible_(false),
        is_input_contiguous_{true},
        is_resource_nice_{true},
        max_batch_size_{0},
        default_batch_size_{0} {};
  virtual ~FlowUnitDesc(){};

  const std::string GetFlowUnitName() { return flowunit_name_; };
  const std::string GetFlowUnitAliasName() { return alias_name_; };
  const std::string GetFlowUnitArgument() { return argument_; };
  const bool IsCollapseAll() {
    if (loop_type_ != LOOP) {
      if (output_type_ != COLLAPSE) {
        return false;
      }
      return is_collapse_all_;
    }

    return true;
  };

  const bool IsStreamSameCount() {
    if (flow_type_ == NORMAL) {
      return true;
    }
    return is_stream_same_count_;
  };

  bool IsInputContiguous() const { return is_input_contiguous_; }

  bool IsResourceNice() const { return is_resource_nice_; }

  const bool IsExceptionVisible() { return is_exception_visible_; };

  const ConditionType GetConditionType() { return condition_type_; };

  const FlowOutputType GetOutputType() { return output_type_; };

  const FlowType GetFlowType() { return flow_type_; };

  const LoopType GetLoopType() { return loop_type_; };

  const std::string GetGroupType() { return group_type_; };

  const uint32_t GetMaxBatchSize() {
    if (max_batch_size_ != 0) {
      return max_batch_size_;
    }

    // return default value
    if (flow_type_ == STREAM) {
      return STREAM_MAX_BATCH_SIZE;
    }
    return NORMAL_MAX_BATCH_SIZE;
  };

  const uint32_t GetDefaultBatchSize() {
    if (default_batch_size_ != 0) {
      return default_batch_size_;
    }

    // return default value
    if (flow_type_ == STREAM) {
      return STREAM_DEFAULT_BATCH_SIZE;
    }
    return NORMAL_DEFAULT_BATCH_SIZE;
  };

  std::vector<FlowUnitInput> &GetFlowUnitInput() {
    return flowunit_input_list_;
  };
  const std::vector<FlowUnitOutput> &GetFlowUnitOutput() {
    return flowunit_output_list_;
  };

  std::vector<FlowUnitOption> &GetFlowUnitOption() {
    return flowunit_option_list_;
  }

  std::shared_ptr<DriverDesc> GetDriverDesc() { return driver_desc_; }

  std::string GetDescription() { return flowunit_description_; }

  std::string GetVirtualType() { return virtual_type_; }

  void SetFlowUnitName(const std::string &flowunit_name);
  Status AddFlowUnitInput(const FlowUnitInput &flowunit_input);
  Status AddFlowUnitOutput(const FlowUnitOutput &flowunit_output);
  Status AddFlowUnitOption(const FlowUnitOption &flowunit_option);

  void SetFlowUnitGroupType(const std::string &group_type) {
    if (CheckGroupType(group_type) != STATUS_SUCCESS) {
      MBLOG_WARN << "check group type failed , your group_type is "
                 << group_type
                 << ", the right group_type is a or a/b , for instance input "
                    "or input/http.";
      return;
    }

    group_type_ = group_type;
  };

  void SetDriverDesc(std::shared_ptr<DriverDesc> driver_desc) {
    driver_desc_ = driver_desc;
  }

  void SetFlowUnitAliasName(const std::string &alias_name) {
    alias_name_ = alias_name;
  };

  void SetFlowUnitArgument(const std::string &argument) {
    argument_ = argument;
  };

  void SetConditionType(ConditionType condition_type) {
    condition_type_ = condition_type;
  }

  void SetLoopType(LoopType loop_type) { loop_type_ = loop_type; }

  void SetOutputType(FlowOutputType output_type) { output_type_ = output_type; }

  void SetFlowType(FlowType flow_type) { flow_type_ = flow_type; }

  void SetStreamSameCount(bool is_stream_same_count) {
    if (flow_type_ == STREAM) {
      is_stream_same_count_ = is_stream_same_count;
    }
  };

  void SetInputContiguous(bool is_input_contiguous) {
    is_input_contiguous_ = is_input_contiguous;
  }

  void SetResourceNice(bool is_resource_nice) {
    is_resource_nice_ = is_resource_nice;
  }

  void SetCollapseAll(bool is_collapse_all) {
    if (output_type_ == COLLAPSE) {
      is_collapse_all_ = is_collapse_all;
    }
  };

  void SetExceptionVisible(bool is_exception_visible) {
    is_exception_visible_ = is_exception_visible;
  };

  void SetVirtualType(const std::string &virtual_type) {
    virtual_type_ = virtual_type;
  }

  void SetDescription(const std::string &description) {
    flowunit_description_ = description;
  }

  void SetMaxBatchSize(const uint32_t &max_batch_size) {
    if (max_batch_size == 0) {
      MBLOG_ERROR << "max_batch_size must be greater than zero.";
      return;
    }
    max_batch_size_ = max_batch_size;
  }

  void SetDefaultBatchSize(const uint32_t &default_batch_size) {
    if (default_batch_size == 0) {
      MBLOG_ERROR << "default_batch_size must be greater than zero.";
      return;
    }
    default_batch_size_ = default_batch_size;
  }

 protected:
  FlowOutputType output_type_;

  FlowType flow_type_;

  ConditionType condition_type_;

  LoopType loop_type_;

  bool is_stream_same_count_;
  bool is_collapse_all_;
  bool is_exception_visible_;
  std::string flowunit_name_;
  std::string group_type_;
  std::string alias_name_;
  std::string argument_;
  std::string virtual_type_;
  std::string flowunit_description_;
  std::vector<FlowUnitInput> flowunit_input_list_;
  std::vector<FlowUnitOutput> flowunit_output_list_;
  std::vector<FlowUnitOption> flowunit_option_list_;
  std::shared_ptr<DriverDesc> driver_desc_;
  bool is_input_contiguous_;
  bool is_resource_nice_;
  uint32_t max_batch_size_;
  uint32_t default_batch_size_;

 private:
  Status CheckInputDuplication(const FlowUnitInput &flowunit_input);
  Status CheckOutputDuplication(const FlowUnitOutput &flowunit_output);
  Status CheckOptionDuplication(const FlowUnitOption &flowunit_option);
  Status CheckGroupType(const std::string &group_type);
};

class FlowUnitInnerEvent;
class ExternalData;

using CreateExternalDataFunc =
    std::function<std::shared_ptr<ExternalData>(std::shared_ptr<Device>)>;

class FlowUnitStreamContext {
 public:
  enum StreamMode { EXPAND_DATA, DEFAULT, COLLAPSE_DATA };

  enum RecvMode { SYNC, ASYNC };

  FlowUnitStreamContext();
  virtual ~FlowUnitStreamContext();

  bool HasError();

  bool HasError(const std::string &port);

  std::shared_ptr<FlowUnitError> GetError();

  std::shared_ptr<FlowUnitError> GetError(const std::string &port);

  bool HasEvent();

  std::shared_ptr<FlowUnitInnerEvent> RecvEvent();

  void SendEvent(std::shared_ptr<FlowUnitInnerEvent> event);

  bool HasExternalData();

  Status SendExternalData(BufferList &buffer);

  Status RecvExternalData(BufferList &buffer);

  Status RecvData(const std::string &port, BufferList &buffer);

  Status SendData(const std::string &port, BufferList &buffer);

  void NewOutputStream(StreamMode type, const std::string &port,
                       std::shared_ptr<DataMeta> data_meta);

  const std::shared_ptr<DataMeta> GetInputMeta(const std::string &port);

  const std::shared_ptr<DataMeta> GetInputGroupMeta(const std::string &port);

  std::shared_ptr<DataMeta> GetOutputMeta(const std::string &port);

  std::shared_ptr<SessionContext> GetSessionContext();

  void SetPrivate(const std::string &key, std::shared_ptr<void> private_value);

  std::shared_ptr<void> GetPrivate(const std::string &key);

  Status CloseAll();

  Status Close(const std::string &port);

  void SetRecvMode(RecvMode recv_mode);

  BufferList &NewBufferList();
};

/**使用说明

class FlowUnitStream {
 public:
  FlowUnitStream();
  virtual ~FlowUnitStream();

  Open();

  Close();

  virtual Status Process(std::shared_ptr<FlowUnitStreamContext> ctx) {
    if (ctx.HasError("IN1|IN2|OUT1|OUT2") != 0) {
      ctx.Close();
      return;
    }

    if (ctx.HasEventData()) {
      processEvent();
    }

    if (ctx.HasExternalData()) {
      processEvent();
    }

    auto ret = ctx.GetExternalData(buffer1);
    auto ret = ctx.RecvData("IN1", buffer1);
    auto ret = ctx.RecvData("IN2", buffer2);

    auto buff = ctx.NewBuffer();
    //process;
    ctx.SendData("OUT1", buff);
    ctx.SendData("OUT2", buff);

    return STATUS_FAULT;
  }

  virtual Status StreamOpen(FlowUnitStreamContext ctx) = 0 {
    auto meta1 = ctx.GetMeta("IN1");
    auto met2a = ctx.GetMeta("IN1");
    cts.SetInRecvMode(MODE_MATCH);
    auto groupmeta = in2.GetInputStreamGroupMeta();
    ctx.NewOutputStream("out1", meta1, EXPAND_DATA);
    ctx.NewOutputStream("out2", meta2, EXPAND_DATA);
  }

  virtual Status StreamClose(FlowUnitStreamContext ctx) = 0 {
    ctx.CloseOutputStream("out1");
    ctx.CloseOutputStream("out2");
  }

  virtual Status StreamGroupOpen(FlowUnitStreamContext ctx) = 0; {
    ctx.NewOutputStream("out2", meta2, COLLAPSE_DATA);
  }

  virtual Status StreamGroupClose(FlowUnitStreamContext ctx) {
    ctx.CloseOutputStream("out2");
  }
};
**/
class FlowUnitStream {
 public:
  FlowUnitStream();
  virtual ~FlowUnitStream();

  virtual Status Open(const std::shared_ptr<Configuration> &config) = 0;

  /* class when unit is close */
  virtual Status Close() = 0;

  virtual Status Process(std::shared_ptr<FlowUnitStreamContext> ctx) = 0;

  virtual Status StreamOpen(
      const std::shared_ptr<FlowUnitStreamContext> ctx) = 0;

  virtual Status StreamClose(
      const std::shared_ptr<FlowUnitStreamContext> ctx) = 0;

  virtual Status ParentStreamOpen(
      const std::shared_ptr<FlowUnitStreamContext> ctx) = 0;

  virtual Status ParentStreamClose(
      const std::shared_ptr<FlowUnitStreamContext> ctx) = 0;
};

/**
 * @brief Flowunit plugin interface
 */
class IFlowUnit {
 public:
  IFlowUnit();
  virtual ~IFlowUnit();

  /**
   * @brief Flowunit open function, called when unit is open for processing data
   * @param config flowunit configuration
   * @return open result
   */
  virtual Status Open(const std::shared_ptr<Configuration> &config) = 0;

  /**
   * @brief Flowunit close function, called when unit is closed.
   * @return open result
   */
  virtual Status Close();

  /**
   * @brief Flowunit data process.
   * @param data_ctx data context.
   * @return open result
   */
  virtual Status Process(std::shared_ptr<DataContext> data_ctx) = 0;

  /**
   * @brief Flowunit data pre.
   * @param data_ctx data context.
   * @return data pre result
   */
  virtual Status DataPre(std::shared_ptr<DataContext> data_ctx);

  /**
   * @brief Flowunit data post.
   * @param data_ctx data context.
   * @return data post result
   */
  virtual Status DataPost(std::shared_ptr<DataContext> data_ctx);

  /**
   * @brief Flowunit data group pre.
   * @param data_ctx data context.
   * @return data group result
   */
  virtual Status DataGroupPre(std::shared_ptr<DataContext> data_ctx);

  /**
   * @brief Flowunit data group post.
   * @param data_ctx data context.
   * @return data group post result
   */
  virtual Status DataGroupPost(std::shared_ptr<DataContext> data_ctx);
};

class FlowUnit : public IFlowUnit {
 public:
  FlowUnit(){};
  virtual ~FlowUnit(){};

  /* called when unit is open for process */
  virtual Status Open(const std::shared_ptr<Configuration> &config) = 0;

  /* class when unit is close */
  virtual Status Close() = 0;

  virtual void SetFlowUnitDesc(std::shared_ptr<FlowUnitDesc> desc);
  virtual std::shared_ptr<FlowUnitDesc> GetFlowUnitDesc();

  void SetBindDevice(std::shared_ptr<Device> device);
  std::shared_ptr<Device> GetBindDevice();

  void SetExternalData(const CreateExternalDataFunc &create_external_data) {
    create_ext_data_func_ = create_external_data;
  }

  std::shared_ptr<ExternalData> CreateExternalData() const {
    if (!create_ext_data_func_) {
      return nullptr;
    }

    return create_ext_data_func_(device_);
  }

 protected:
  CreateExternalDataFunc GetCreateExternalDataFunc() {
    return create_ext_data_func_;
  }

  int32_t dev_id_{0};

 private:
  std::shared_ptr<FlowUnitDesc> flowunit_desc_ =
      std::make_shared<FlowUnitDesc>();
  std::shared_ptr<Device> device_;

  CreateExternalDataFunc create_ext_data_func_;
};

class FlowUnitFactory : public DriverFactory {
 public:
  FlowUnitFactory(){};
  virtual ~FlowUnitFactory(){};

  virtual std::map<std::string, std::shared_ptr<FlowUnitDesc>> FlowUnitProbe() {
    return std::map<std::string, std::shared_ptr<FlowUnitDesc>>();
  };

  virtual void SetDriver(std::shared_ptr<Driver> driver) override {
    driver_ = driver;
  }

  virtual std::shared_ptr<Driver> GetDriver() override { return driver_; }

  virtual const std::string GetFlowUnitFactoryType() { return ""; };

  virtual const std::string GetFlowUnitFactoryName() { return ""; };
  virtual const std::vector<std::string> GetFlowUnitNames() {
    return std::vector<std::string>();
  };

  virtual const std::string GetVirtualType() { return ""; };
  virtual void SetVirtualType(const std::string &virtual_type) { return; };

  virtual std::shared_ptr<FlowUnit> CreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type) {
    if (GetVirtualType().empty()) {
      StatusError = {STATUS_FAULT, "invalid Flow Unit"};
      return nullptr;
    }

    return VirtualCreateFlowUnit(unit_name, unit_type, GetVirtualType());
  };

  virtual std::shared_ptr<FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type) {
    StatusError = {STATUS_FAULT, "Invalid virtual flowunit"};
    return nullptr;
  }

  virtual void SetFlowUnitFactory(
      std::vector<std::shared_ptr<modelbox::DriverFactory>>
          bind_flowunit_factory_list) {
    return;
  };

 private:
  std::shared_ptr<Driver> driver_;
};

using FlowUnitDeviceConfig =
    std::unordered_map<std::string, std::vector<std::string>>;

class FlowUnitManager {
 public:
  FlowUnitManager();
  virtual ~FlowUnitManager();

  static std::shared_ptr<FlowUnitManager> GetInstance();

  Status Register(std::shared_ptr<FlowUnitFactory> factory);

  Status Initialize(std::shared_ptr<Drivers> driver,
                    std::shared_ptr<DeviceManager> device_mgr,
                    std::shared_ptr<Configuration> config);

  virtual std::vector<std::string> GetFlowUnitTypes();

  virtual std::vector<std::string> GetFlowUnitList(
      const std::string &unit_type);

  virtual std::vector<std::string> GetFlowUnitTypes(
      const std::string &unit_name);

  std::vector<std::shared_ptr<FlowUnit>> CreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type = "",
      const std::string &unit_device_id = "");

  Status FlowUnitProbe();
  Status InitFlowUnitFactory(std::shared_ptr<Drivers> driver);
  Status SetUpFlowUnitDesc();
  void Clear();
  /**
   * GetFlowUnitFactoryList(), GetFlowUnitDescList()
   * only for test
   */

  std::map<std::pair<std::string, std::string>,
           std::shared_ptr<FlowUnitFactory>>
  GetFlowUnitFactoryList();
  std::map<std::string, std::map<std::string, std::shared_ptr<FlowUnitDesc>>>
  GetFlowUnitDescList();

  void InsertFlowUnitFactory(
      const std::string &name, const std::string &type,
      const std::shared_ptr<FlowUnitFactory> &flowunit_factory);
      
  std::vector<std::shared_ptr<FlowUnitDesc>> GetAllFlowUnitDesc();

  std::shared_ptr<FlowUnitDesc> GetFlowUnitDesc(
      const std::string &flowunit_type, const std::string &flowunit_name);

  std::shared_ptr<DeviceManager> GetDeviceManager();

 private:
  modelbox::Status CheckParams(const std::string &unit_name,
                               const std::string &unit_type,
                               const std::string &unit_device_id);

  modelbox::Status ParseUnitDeviceConf(const std::string &unit_name,
                                       const std::string &unit_type,
                                       const std::string &unit_device_id,
                                       FlowUnitDeviceConfig &dev_cfg);

  modelbox::Status ParseUserDeviceConf(const std::string &unit_type,
                                       const std::string &unit_device_id,
                                       FlowUnitDeviceConfig &dev_cfg);

  modelbox::Status AutoFillDeviceConf(const std::string &unit_name,
                                      FlowUnitDeviceConfig &dev_cfg);

  void SetDeviceManager(std::shared_ptr<DeviceManager> device_mgr);
  std::shared_ptr<FlowUnit> CreateSingleFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &unit_device_id);
  std::shared_ptr<DeviceManager> device_mgr_;
  std::map<std::pair<std::string, std::string>,
           std::shared_ptr<FlowUnitFactory>>
      flowunit_factory_;

  std::map<std::string, std::map<std::string, std::shared_ptr<FlowUnitDesc>>>
      flowunit_desc_list_;
};
}  // namespace modelbox
#endif  // MODELBOX_FLOW_UNIT_H_
