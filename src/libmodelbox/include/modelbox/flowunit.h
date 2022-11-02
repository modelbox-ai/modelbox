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
#include <utility>
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
  FlowUnitPort(std::string name);
  FlowUnitPort(std::string name, std::string device_type);
  FlowUnitPort(std::string name, uint32_t device_mem_flags);
  FlowUnitPort(std::string name, std::string device_type,
               uint32_t device_mem_flags);
  FlowUnitPort(std::string name, std::string device_type, std::string type);
  FlowUnitPort(std::string name, std::string device_type, std::string type,
               std::map<std::string, std::string> ext);

  virtual ~FlowUnitPort();

  void SetDeviceType(const std::string &device_type);

  void SetPortName(const std::string &port_name);

  void SetPortType(const std::string &port_type);

  void SetDevice(std::shared_ptr<Device> device);

  void SetProperity(const std::string &key, const std::string &value);

  std::string GetDeviceType() const;

  std::string GetPortName() const;

  std::string GetPortType() const;

  std::shared_ptr<Device> GetDevice() const;

  uint32_t GetDeviceMemFlags() const;

  std::string GetProperity(const std::string &key);

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
  FlowUnitInput(const std::string &name);
  FlowUnitInput(const std::string &name, const std::string &device_type);
  FlowUnitInput(const std::string &name, uint32_t device_mem_flags);
  FlowUnitInput(const std::string &name, const std::string &device_type,
                uint32_t device_mem_flags);
  FlowUnitInput(const std::string &name, const std::string &device_type,
                const std::string &type);
  FlowUnitInput(const std::string &name, const std::string &device_type,
                const std::string &type,
                const std::map<std::string, std::string> &ext);
  ~FlowUnitInput() override;
};

class FlowUnitOutput : public FlowUnitPort {
 public:
  FlowUnitOutput(const std::string &name);
  FlowUnitOutput(const std::string &name, uint32_t device_mem_flags);
  /**
   * @deprecated
   **/
  FlowUnitOutput(const std::string &name, const std::string &device_type);
  /**
   * @deprecated
   **/
  FlowUnitOutput(const std::string &name, const std::string &device_type,
                 uint32_t device_mem_flags);
  /**
   * @deprecated
   **/
  FlowUnitOutput(const std::string &name, const std::string &device_type,
                 const std::string &type);
  /**
   * @deprecated
   **/
  FlowUnitOutput(const std::string &name, const std::string &device_type,
                 const std::string &type,
                 const std::map<std::string, std::string> &ext);
  ~FlowUnitOutput() override;
};

class FlowUnitOption {
 public:
  FlowUnitOption(std::string name, std::string type);
  FlowUnitOption(std::string name, std::string type, bool require);
  FlowUnitOption(std::string name, std::string type, bool require,
                 std::string default_value, std::string desc,
                 std::map<std::string, std::string> values);
  FlowUnitOption(std::string name, std::string type, bool require,
                 std::string default_value, std::string desc);

  virtual ~FlowUnitOption();

  void SetOptionName(const std::string &option_name);

  void SetOptionType(const std::string &option_type);

  void SetOptionRequire(bool option_require);

  void SetOptionDesc(const std::string &option_desc);

  void AddOptionValue(const std::string &key, const std::string &value);

  std::string GetOptionName() const;

  std::string GetOptionType() const;

  bool IsRequire() const;

  std::string GetOptionDefault() const;

  std::string GetOptionDesc() const;

  std::map<std::string, std::string> GetOptionValues();

  std::string GetOptionValue(const std::string &key);

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
  FlowUnitDesc();
  virtual ~FlowUnitDesc();

  std::string GetFlowUnitName();

  std::string GetFlowUnitType();

  std::string GetFlowUnitAliasName();

  std::string GetFlowUnitArgument();

  bool IsCollapseAll();

  bool IsStreamSameCount();

  bool IsInputContiguous() const;

  bool IsResourceNice() const;

  bool IsExceptionVisible();

  ConditionType GetConditionType();

  FlowOutputType GetOutputType();

  bool IsUserSetFlowType();

  FlowType GetFlowType();

  LoopType GetLoopType();

  std::string GetGroupType();

  uint32_t GetMaxBatchSize();

  uint32_t GetDefaultBatchSize();

  std::vector<FlowUnitInput> &GetFlowUnitInput();

  const std::vector<FlowUnitOutput> &GetFlowUnitOutput();

  std::vector<FlowUnitOption> &GetFlowUnitOption();

  std::shared_ptr<DriverDesc> GetDriverDesc();

  std::string GetDescription();

  std::string GetVirtualType();

  void SetFlowUnitName(const std::string &flowunit_name);

  void SetFlowUnitType(const std::string &flowunit_type);

  Status AddFlowUnitInput(const FlowUnitInput &flowunit_input);

  Status AddFlowUnitOutput(const FlowUnitOutput &flowunit_output);

  Status AddFlowUnitOption(const FlowUnitOption &flowunit_option);

  void SetFlowUnitGroupType(const std::string &group_type);

  void SetDriverDesc(std::shared_ptr<DriverDesc> driver_desc);

  void SetFlowUnitAliasName(const std::string &alias_name);

  void SetFlowUnitArgument(const std::string &argument);

  void SetConditionType(ConditionType condition_type);

  void SetLoopType(LoopType loop_type);

  void SetOutputType(FlowOutputType output_type);

  void SetFlowType(FlowType flow_type);

  void SetStreamSameCount(bool is_stream_same_count);

  void SetInputContiguous(bool is_input_contiguous);

  void SetResourceNice(bool is_resource_nice);
  void SetCollapseAll(bool is_collapse_all);

  void SetExceptionVisible(bool is_exception_visible);

  void SetVirtualType(const std::string &virtual_type);

  void SetDescription(const std::string &description);

  void SetMaxBatchSize(const uint32_t &max_batch_size);

  void SetDefaultBatchSize(const uint32_t &default_batch_size);

 protected:
  FlowOutputType output_type_{ORIGIN};

  bool is_user_set_flow_type_{false};
  FlowType flow_type_{NORMAL};

  ConditionType condition_type_{NONE};

  LoopType loop_type_{NOT_LOOP};

  bool is_stream_same_count_{false};
  bool is_collapse_all_{true};
  bool is_exception_visible_{false};
  std::string flowunit_name_;
  std::string flowunit_type_;
  std::string group_type_;
  std::string alias_name_;
  std::string argument_;
  std::string virtual_type_;
  std::string flowunit_description_;
  std::vector<FlowUnitInput> flowunit_input_list_;
  std::vector<FlowUnitOutput> flowunit_output_list_;
  std::vector<FlowUnitOption> flowunit_option_list_;
  std::shared_ptr<DriverDesc> driver_desc_;
  bool is_input_contiguous_{true};
  bool is_resource_nice_{true};
  uint32_t max_batch_size_{0};
  uint32_t default_batch_size_{0};

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

  template <typename T>
  inline std::shared_ptr<T> GetPrivate(const std::string &key) {
    return std::static_pointer_cast<T>(GetPrivate(key));
  }

  Status CloseAll();

  Status Close(const std::string &port);

  void SetRecvMode(RecvMode recv_mode);

  BufferList &NewBufferList();
};

class FlowUnitStream {
 public:
  FlowUnitStream();
  virtual ~FlowUnitStream();

  virtual Status Open(const std::shared_ptr<Configuration> &config) = 0;

  /* class when unit is close */
  virtual Status Close() = 0;

  virtual Status Process(std::shared_ptr<FlowUnitStreamContext> ctx) = 0;

  virtual Status StreamOpen(std::shared_ptr<FlowUnitStreamContext> ctx) = 0;

  virtual Status StreamClose(std::shared_ptr<FlowUnitStreamContext> ctx) = 0;

  virtual Status ParentStreamOpen(
      std::shared_ptr<FlowUnitStreamContext> ctx) = 0;

  virtual Status ParentStreamClose(
      std::shared_ptr<FlowUnitStreamContext> ctx) = 0;
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
  // NOLINTNEXTLINE
  virtual Status Process(std::shared_ptr<DataContext> data_ctx) = 0;

  /**
   * @brief Flowunit data pre.
   * @param data_ctx data context.
   * @return data pre result
   */
  // NOLINTNEXTLINE
  virtual Status DataPre(std::shared_ptr<DataContext> data_ctx);

  /**
   * @brief Flowunit data post.
   * @param data_ctx data context.
   * @return data post result
   */
  // NOLINTNEXTLINE
  virtual Status DataPost(std::shared_ptr<DataContext> data_ctx);

  /**
   * @deprecated
   * @brief Flowunit data group pre.
   * @param data_ctx data context.
   * @return data group result
   */
  // NOLINTNEXTLINE
  virtual Status DataGroupPre(std::shared_ptr<DataContext> data_ctx);

  /**
   * @deprecated
   * @brief Flowunit data group post.
   * @param data_ctx data context.
   * @return data group post result
   */
  // NOLINTNEXTLINE
  virtual Status DataGroupPost(std::shared_ptr<DataContext> data_ctx);
};

class FlowUnit : public IFlowUnit {
 public:
  FlowUnit();
  ~FlowUnit() override;

  /* called when unit is open for process */
  Status Open(const std::shared_ptr<Configuration> &config) override;

  /* class when unit is close */
  Status Close() override;

  virtual void SetFlowUnitDesc(std::shared_ptr<FlowUnitDesc> desc);

  virtual std::shared_ptr<FlowUnitDesc> GetFlowUnitDesc();

  void SetBindDevice(const std::shared_ptr<Device> &device);

  std::shared_ptr<Device> GetBindDevice();

  void SetExternalData(const CreateExternalDataFunc &create_external_data);

  std::shared_ptr<ExternalData> CreateExternalData() const;

 protected:
  CreateExternalDataFunc GetCreateExternalDataFunc();
  int32_t dev_id_{0};

 private:
  std::shared_ptr<FlowUnitDesc> flowunit_desc_ =
      std::make_shared<FlowUnitDesc>();
  std::shared_ptr<Device> device_;

  CreateExternalDataFunc create_ext_data_func_;
};

class FlowUnitFactory : public DriverFactory {
 public:
  FlowUnitFactory();
  ~FlowUnitFactory() override;

  virtual std::map<std::string, std::shared_ptr<FlowUnitDesc>> FlowUnitProbe();

  void SetDriver(const std::shared_ptr<Driver> &driver) override;

  std::shared_ptr<Driver> GetDriver() override;

  virtual std::string GetFlowUnitFactoryType();

  virtual std::string GetFlowUnitFactoryName();

  virtual std::vector<std::string> GetFlowUnitNames();

  virtual std::string GetVirtualType();

  virtual void SetVirtualType(const std::string &virtual_type);

  virtual std::string GetFlowUnitInputDeviceType();

  virtual std::shared_ptr<FlowUnit> CreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type);

  virtual std::shared_ptr<FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type);

  virtual void SetFlowUnitFactory(
      const std::vector<std::shared_ptr<DriverFactory>>
          &bind_flowunit_factory_list);

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

  Status Register(const std::shared_ptr<FlowUnitFactory> &factory);

  Status Initialize(const std::shared_ptr<Drivers> &driver,
                    std::shared_ptr<DeviceManager> device_mgr,
                    const std::shared_ptr<Configuration> &config);

  virtual std::vector<std::string> GetFlowUnitTypes();

  virtual std::vector<std::string> GetFlowUnitList(
      const std::string &unit_type);

  virtual std::vector<std::string> GetFlowUnitTypes(
      const std::string &unit_name);

  std::vector<std::shared_ptr<FlowUnit>> CreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type = "",
      const std::string &unit_device_id = "");

  Status FlowUnitProbe();
  Status InitFlowUnitFactory(const std::shared_ptr<Drivers> &driver);
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

  std::vector<std::shared_ptr<FlowUnitDesc>> GetAllFlowUnitDesc();

  std::shared_ptr<FlowUnitDesc> GetFlowUnitDesc(
      const std::string &flowunit_type, const std::string &flowunit_name);

  std::shared_ptr<DeviceManager> GetDeviceManager();

 private:
  Status CheckParams(const std::string &unit_name, const std::string &unit_type,
                     const std::string &unit_device_id);

  Status ParseUnitDeviceConf(const std::string &unit_name,
                             const std::string &unit_type,
                             const std::string &unit_device_id,
                             FlowUnitDeviceConfig &dev_cfg);

  Status ParseUserDeviceConf(const std::string &unit_type,
                             const std::string &unit_device_id,
                             FlowUnitDeviceConfig &dev_cfg);

  Status AutoFillDeviceConf(const std::string &unit_name,
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
