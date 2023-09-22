
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

#include "modelbox_api.h"

#include <modelbox/base/config.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <securec.h>

#include <string>
#include <utility>

#include "modelbox/data_context.h"
#include "modelbox/error.h"
#include "modelbox/external_data_simple.h"
#include "modelbox/flow.h"
#include "modelbox/modelbox_engine.h"
#include "modelbox/type.h"
#include "python_common.h"
#include "python_flow.h"
#include "python_log.h"
#include "python_model.h"

constexpr int NPY_FLOAT16 = 23;
template <>
struct pybind11::detail::npy_format_descriptor<modelbox::Float16> {
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }

  static std::string format() {
    // following:
    // https://docs.python.org/3/library/struct.html#format-characters
    return "e";
  }

  static constexpr auto name() -> pybind11::detail::descr<7> {
    return _("float16");
  }
};

namespace modelbox {

class NumpyInfo {
 public:
  NumpyInfo(ssize_t itemsize, const std::string &format, void *ptr,
            std::vector<ssize_t> shape, std::vector<ssize_t> strides)
      : shape_(std::move(shape)),
        strides_(std::move(strides)),
        itemsize_(itemsize) {
    ssize_t bytes = std::accumulate(shape_.begin(), shape_.end(), (ssize_t)1,
                                    std::multiplies<ssize_t>()) *
                    itemsize_;
    m_data_ = std::unique_ptr<char[]>(new char[bytes]);
    memcpy_s(m_data_.get(), bytes, ptr, bytes);
    dtype_ = TypeFromFormatStr(format);
  }

  NumpyInfo(const NumpyInfo &obj) {
    shape_ = obj.Shape();
    strides_ = obj.Strides();
    itemsize_ = obj.ItemSize();
    ssize_t bytes = std::accumulate(shape_.begin(), shape_.end(), (ssize_t)1,
                                    std::multiplies<ssize_t>()) *
                    itemsize_;
    m_data_ = std::unique_ptr<char[]>(new char[bytes]);
    memcpy_s(m_data_.get(), bytes, obj.Data(), bytes);
    dtype_ = obj.Type();
  }
  modelbox::ModelBoxDataType Type() const { return dtype_; }
  std::vector<ssize_t> Shape() const { return shape_; }
  std::vector<ssize_t> Strides() const { return strides_; }
  void *Data() const { return (void *)m_data_.get(); }
  std::size_t ItemSize() const { return itemsize_; }

 private:
  std::vector<ssize_t> shape_;
  std::vector<ssize_t> strides_;
  modelbox::ModelBoxDataType dtype_;
  ssize_t itemsize_;
  std::unique_ptr<char[]> m_data_;
};

template <typename DataType, typename PyType, typename CType>
bool DataSet(DataType &data, const std::string &key, py::object set_obj,
             py::object cast_obj) {
  if (!py::isinstance<PyType>(set_obj)) {
    return false;
  }
  if (auto *buffer = dynamic_cast<Buffer *>(&data)) {
    buffer->Set(key, cast_obj.cast<CType>());
  } else if (auto *session_context = dynamic_cast<SessionContext *>(&data)) {
    auto c_context = std::make_shared<CType>(cast_obj.cast<CType>());
    auto c_type = typeid(CType).hash_code();
    session_context->SetPrivate(key, c_context, c_type);
  } else {
    return false;
  }
  return true;
}

typedef bool (*pDataGetFunc)(std::size_t hash_code, void *, py::object &ret);

template <typename CType, typename PyType>
bool DataGet(std::size_t hash_code, void *value, py::object &ret) {
  if (typeid(CType).hash_code() == hash_code) {
    ret = py::cast(*((CType *)(value)));
    return true;
  }
  return false;
}

template <typename DataType, typename FuncType>
bool SetAttributes(DataType &context, const std::string &key,
                   const py::object &obj, std::vector<FuncType> &BaseObjectFunc,
                   std::vector<FuncType> &List1DObjectFunc,
                   std::vector<FuncType> &List2DObjectFunc) {
  auto setup_data = [&](std::vector<FuncType> &func_list,
                        const py::object &set_obj, const py::object &cast_obj) {
    for (auto &func : func_list) {
      if (func(context, key, set_obj, cast_obj)) {
        return true;
      }
    }
    return false;
  };

  if (setup_data(BaseObjectFunc, obj, obj)) {
    return true;
  }

  if (py::isinstance<py::list>(obj)) {
    py::list obj_list_all = obj.cast<py::list>();
    if (obj_list_all.empty()) {
      return setup_data(List1DObjectFunc, obj_list_all, obj_list_all);
    }
    if (py::isinstance<py::list>(obj_list_all[0])) {
      py::list obj_list_1d = obj_list_all[0].cast<py::list>();
      if (obj_list_1d.empty()) {
        return setup_data(List2DObjectFunc, obj_list_1d, obj_list_all);
      }
      if (setup_data(List2DObjectFunc, obj_list_1d[0], obj_list_all)) {
        return true;
      }
    } else {
      if (setup_data(List1DObjectFunc, obj_list_all[0], obj_list_all)) {
        return true;
      }
    }
  }
  return false;
}

template <typename FuncType>
bool GetAttributes(void *value, std::size_t value_type,
                   std::vector<FuncType> func_list, py::object &ret_data) {
  for (auto &pfunc : func_list) {
    if (pfunc(value_type, value, ret_data)) {
      return true;
    }
  }

  if (typeid(NumpyInfo).hash_code() == value_type) {
    auto *data = (NumpyInfo *)(value);
    if (data == nullptr) {
      MBLOG_ERROR << "data is nullptr.";
    }

    auto buffer_info =
        py::buffer_info((void *)(data->Data()), data->ItemSize(),
                        FormatStrFromType(data->Type()), data->Shape().size(),
                        data->Shape(), data->Strides());
    ret_data = py::array(buffer_info);
    return true;
  }
  if (typeid(py::object).hash_code() == value_type) {
    ret_data = *((py::object *)(value));
    return true;
  }

  if (typeid(std::shared_ptr<py::object>).hash_code() == value_type) {
    ret_data = *(*((std::shared_ptr<py::object> *)(value)));
    return true;
  }

  return false;
}

/*******************************BufferSet Begin**************************/
typedef bool (*pBufferPyTypeToCTypeFunc)(Buffer &, const std::string &,
                                         py::object, py::object);
static std::vector<pBufferPyTypeToCTypeFunc> kBufferBaseObjectFunc = {
    DataSet<Buffer, py::float_, double>,
    DataSet<Buffer, ModelBoxDataType, ModelBoxDataType>,
    DataSet<Buffer, py::str, std::string>, DataSet<Buffer, py::bool_, bool>,
    DataSet<Buffer, py::int_, long>};

static std::vector<pBufferPyTypeToCTypeFunc> kBufferLIst1DObjectFunc = {
    DataSet<Buffer, py::float_, std::vector<double>>,
    DataSet<Buffer, py::str, std::vector<std::string>>,
    DataSet<Buffer, py::bool_, std::vector<bool>>,
    DataSet<Buffer, py::int_, std::vector<long>>};

static std::vector<pBufferPyTypeToCTypeFunc> kBufferLIst2DObjectFunc = {
    DataSet<Buffer, py::float_, std::vector<std::vector<double>>>,
    DataSet<Buffer, py::str, std::vector<std::vector<std::string>>>,
    DataSet<Buffer, py::bool_, std::vector<std::vector<bool>>>,
    DataSet<Buffer, py::int_, std::vector<std::vector<long>>>};

void BufferSetAttributes(Buffer &buffer, const std::string &key,
                         py::object &obj) {
  if (SetAttributes<Buffer, pBufferPyTypeToCTypeFunc>(
          buffer, key, obj, kBufferBaseObjectFunc, kBufferLIst1DObjectFunc,
          kBufferLIst2DObjectFunc)) {
    return;
  }
  if (py::isinstance<py::buffer>(obj)) {
    py::buffer obj_buffer = obj.cast<py::buffer>();
    py::buffer_info buffer_info = obj_buffer.request();
    NumpyInfo numpy_info(buffer_info.itemsize, buffer_info.format,
                         buffer_info.ptr, buffer_info.shape,
                         buffer_info.strides);
    buffer.Set(key, numpy_info);
    return;
  }

  if (py::isinstance<py::object>(obj)) {
    auto *obj_ptr = new py::object();
    *obj_ptr = obj;
    auto obj_shared = std::shared_ptr<py::object>(obj_ptr, [](void *ptr) {
      py::gil_scoped_acquire interpreter_guard{};
      delete static_cast<py::object *>(ptr);
    });
    buffer.Set(key, obj_shared);
    return;
  }

  throw std::invalid_argument("invalid data type " +
                              py::str(obj).cast<std::string>() + " for key " +
                              key);
}
/*******************************BufferSet End*******************************/

/*******************************BufferGet Begin*****************************/

static std::vector<pDataGetFunc> kBufferObjectConvertFunc = {
    DataGet<ModelBoxDataType, ModelBoxDataType>,
    DataGet<int, py::int_>,
    DataGet<unsigned int, py::int_>,
    DataGet<long, py::int_>,
    DataGet<unsigned long, py::int_>,
    DataGet<char, py::int_>,
    DataGet<unsigned char, py::int_>,
    DataGet<float, py::float_>,
    DataGet<double, py::float_>,
    DataGet<std::string, py::str>,
    DataGet<bool, py::bool_>,

    DataGet<std::vector<int>, py::list>,
    DataGet<std::vector<unsigned int>, py::list>,
    DataGet<std::vector<long>, py::list>,
    DataGet<std::vector<unsigned long>, py::list>,
    DataGet<std::vector<char>, py::list>,
    DataGet<std::vector<unsigned char>, py::list>,
    DataGet<std::vector<float>, py::list>,
    DataGet<std::vector<double>, py::list>,
    DataGet<std::vector<std::string>, py::list>,
    DataGet<std::vector<bool>, py::list>,

    DataGet<std::vector<std::vector<float>>, py::list>,
    DataGet<std::vector<std::vector<double>>, py::list>,
    DataGet<std::vector<std::vector<int>>, py::list>,
    DataGet<std::vector<std::vector<unsigned int>>, py::list>,
    DataGet<std::vector<std::vector<long>>, py::list>,
    DataGet<std::vector<std::vector<unsigned long>>, py::list>,
    DataGet<std::vector<std::vector<std::string>>, py::list>,
    DataGet<std::vector<std::vector<bool>>, py::list>,
};

py::object BufferGetAttributes(Buffer &buffer, const std::string &key) {
  auto ret = buffer.Get(key);
  if (!std::get<1>(ret)) {
    throw std::invalid_argument("can not find buffer meta: " + key);
  }
  auto *data = std::get<0>(ret);
  auto value_type = (std::size_t)(data->type().hash_code());
  auto *value = any_cast<void *>(data);
  py::object ret_data;

  if (GetAttributes(value, value_type, kBufferObjectConvertFunc, ret_data)) {
    return ret_data;
  }
  throw std::invalid_argument("invalid data type " +
                              std::string(data->type().name()) +
                              " for buffer meta " + key);
}
/*******************************BufferGet End*****************************/

/***************************ConfigurationSet Begin************************/
typedef bool (*pConfigurationPyTypeToCTypeFunc)(Configuration &config,
                                                const std::string &key,
                                                py::object set_obj,
                                                py::object cast_obj);

template <typename PyType, typename CType>
bool ConfigSet(Configuration &config, const std::string &key,
               py::object set_obj, py::object cast_obj) {
  if (py::isinstance<PyType>(set_obj)) {
    config.SetProperty(key, cast_obj.cast<CType>());
    return true;
  }

  return false;
}

static std::vector<pConfigurationPyTypeToCTypeFunc>
    kConfigurationBaseObjectFunc = {
        ConfigSet<py::float_, double>, ConfigSet<py::str, std::string>,
        ConfigSet<py::bool_, bool>, ConfigSet<py::int_, long>};

static std::vector<pConfigurationPyTypeToCTypeFunc>
    kConfigurationListObjectFunc = {
        ConfigSet<py::float_, std::vector<double>>,
        ConfigSet<py::str, std::vector<std::string>>,
        ConfigSet<py::bool_, std::vector<bool>>,
        ConfigSet<py::int_, std::vector<long>>};
static std::vector<pConfigurationPyTypeToCTypeFunc>
    kConfigurationList2DObjectFunc = {};
void ConfigurationSetAttributes(Configuration &config, const std::string &key,
                                const py::object &obj) {
  if (SetAttributes<Configuration, pConfigurationPyTypeToCTypeFunc>(
          config, key, obj, kConfigurationBaseObjectFunc,
          kConfigurationListObjectFunc, kConfigurationList2DObjectFunc)) {
    return;
  }
  throw std::invalid_argument("invalid data type " +
                              py::str(obj).cast<std::string>() + " for key " +
                              key);
}
/***************************ConfigurationSet End************************/

/***************************DataContextSet Begin************************/
void DataContextSetAttributes(DataContext &data_context, const std::string &key,
                              const py::object &obj) {
  obj.inc_ref();
  auto py_context = std::make_shared<py::object>(obj);
  data_context.SetPrivate(key, py_context);
}
/***************************DataContextSet End**************************/

/***************************DataContextGet Begin************************/
py::object DataContextGetAttributes(DataContext &data_context,
                                    const std::string &key) {
  auto *value = data_context.GetPrivate(key).get();
  return *((py::object *)(value));
}
/***************************DataContextGet End*****************************/

/***************************SessionContextSet Start************************/

typedef bool (*pSessionContextPyTypeToCTypeFunc)(SessionContext &,
                                                 const std::string &,
                                                 py::object, py::object);

static std::vector<pSessionContextPyTypeToCTypeFunc>
    kSessionContextBaseObjectFunc = {
        DataSet<SessionContext, py::float_, double>,
        DataSet<SessionContext, py::str, std::string>,
        DataSet<SessionContext, py::bool_, bool>,
        DataSet<SessionContext, py::int_, long>};

static std::vector<pSessionContextPyTypeToCTypeFunc>
    kSessionContextList1DObjectFunc = {
        DataSet<SessionContext, py::float_, std::vector<double>>,
        DataSet<SessionContext, py::str, std::vector<std::string>>,
        DataSet<SessionContext, py::bool_, std::vector<bool>>,
        DataSet<SessionContext, py::int_, std::vector<long>>};

static std::vector<pSessionContextPyTypeToCTypeFunc>
    kSessionContextList2DObjectFunc = {
        DataSet<SessionContext, py::float_, std::vector<std::vector<double>>>,
        DataSet<SessionContext, py::str, std::vector<std::vector<std::string>>>,
        DataSet<SessionContext, py::bool_, std::vector<std::vector<bool>>>,
        DataSet<SessionContext, py::int_, std::vector<std::vector<long>>>};

void SessionContextSetAttributes(SessionContext &session_context,
                                 const std::string &key,
                                 const py::object &obj) {
  if (SetAttributes<SessionContext, pSessionContextPyTypeToCTypeFunc>(
          session_context, key, obj, kSessionContextBaseObjectFunc,
          kSessionContextList1DObjectFunc, kSessionContextList2DObjectFunc)) {
    return;
  }
  if (py::isinstance<py::buffer>(obj)) {
    py::buffer obj_buffer = obj.cast<py::buffer>();
    auto buffer_info = obj_buffer.request();
    auto buffer_context = std::make_shared<NumpyInfo>(
        buffer_info.itemsize, buffer_info.format, buffer_info.ptr,
        buffer_info.shape, buffer_info.strides);
    auto buffer_type = typeid(NumpyInfo).hash_code();
    session_context.SetPrivate(key, buffer_context, buffer_type);
    return;
  }
  obj.inc_ref();
  auto py_context = std::make_shared<py::object>(obj);
  auto py_type = typeid(py::object).hash_code();
  session_context.SetPrivate(key, py_context, py_type);
}
/***************************SessionContextSet End**************************/

/***************************SessionContextGet Start************************/

static std::vector<pDataGetFunc> kSessionContextObjectConvertFunc =
    kBufferObjectConvertFunc;
py::object SessionContextGetAttributes(SessionContext &session_context,
                                       const std::string &key) {
  py::object ret_data;
  auto *value = session_context.GetPrivate(key).get();
  auto value_type = session_context.GetPrivateType(key);
  if (GetAttributes(value, value_type, kSessionContextObjectConvertFunc,
                    ret_data)) {
    return ret_data;
  }
  throw std::invalid_argument("invalid data type " + key);
}

/***************************SessionContextGet End************************/

void ModelboxPyApiSetUpLog(pybind11::module &m) {
  m.def("set_log_level", FlowUnitPythonLog::SetLogLevel);
  m.def("debug", FlowUnitPythonLog::Debug);
  m.def("info", FlowUnitPythonLog::Info);
  m.def("notice", FlowUnitPythonLog::Notice);
  m.def("warn", FlowUnitPythonLog::Warn);
  m.def("error", FlowUnitPythonLog::Error);
  m.def("fatal", FlowUnitPythonLog::Fatal);
}

void ModelboxPyApiSetUpLogLevel(pybind11::handle &h) {
  py::enum_<LogLevel>(h, "Level", py::arithmetic(), py::module_local())
      .value("DEBUG", LOG_DEBUG)
      .value("INFO", LOG_INFO)
      .value("NOTICE", LOG_NOTICE)
      .value("WARN", LOG_WARN)
      .value("ERROR", LOG_ERROR)
      .value("FATAL", LOG_FATAL)
      .value("OFF", LOG_OFF);
}

void ModelboxPyApiSetUpStatus(pybind11::module &m) {
  auto c =
      py::class_<modelbox::Status, std::shared_ptr<modelbox::Status>>(
          m, "Status", py::module_local())
          .def(py::init<>())
          .def(py::init<const StatusCode &>())
          .def(py::init<const bool &>())
          .def(py::init<const StatusCode &, const std::string &>())
          .def(py::init<const Status &, const std::string &>())
          .def("__str__", &modelbox::Status::ToString)
          .def("__bool__", [](const modelbox::Status &s) { return bool(s); })
          .def(py::self == true)
          .def(py::self == py::self)  // NOLINT
          .def(py::self == StatusCode())
          .def(py::self != py::self)  // NOLINT
          .def(py::self != StatusCode())
          .def("code", &modelbox::Status::Code)
          .def("set_errormsg", &modelbox::Status::SetErrormsg)
          .def("str_code", &modelbox::Status::StrCode)
          .def("errormsg", &modelbox::Status::Errormsg)
          .def("wrap_errormsgs", &modelbox::Status::WrapErrormsgs)
          .def("unwrap", &modelbox::Status::Unwrap);

  py::enum_<StatusCode>(c, "StatusCode", py::arithmetic(), py::module_local())
      .value("STATUS_SUCCESS", STATUS_SUCCESS)
      .value("STATUS_FAULT", STATUS_FAULT)
      .value("STATUS_NOTFOUND", STATUS_NOTFOUND)
      .value("STATUS_INVALID", STATUS_INVALID)
      .value("STATUS_AGAIN", STATUS_AGAIN)
      .value("STATUS_BADCONF", STATUS_BADCONF)
      .value("STATUS_NOMEM", STATUS_NOMEM)
      .value("STATUS_RANGE", STATUS_RANGE)
      .value("STATUS_EXIST", STATUS_EXIST)
      .value("STATUS_INTERNAL", STATUS_INTERNAL)
      .value("STATUS_BUSY", STATUS_BUSY)
      .value("STATUS_PERMIT", STATUS_PERMIT)
      .value("STATUS_NOTSUPPORT", STATUS_NOTSUPPORT)
      .value("STATUS_NODATA", STATUS_NODATA)
      .value("STATUS_NOSPACE", STATUS_NOSPACE)
      .value("STATUS_NOBUFS", STATUS_NOBUFS)
      .value("STATUS_OVERFLOW", STATUS_OVERFLOW)
      .value("STATUS_INPROGRESS", STATUS_INPROGRESS)
      .value("STATUS_ALREADY", STATUS_ALREADY)
      .value("STATUS_TIMEDOUT", STATUS_TIMEDOUT)
      .value("STATUS_NOSTREAM", STATUS_NOSTREAM)
      .value("STATUS_RESET", STATUS_RESET)
      .value("STATUS_CONTINUE", STATUS_CONTINUE)
      .value("STATUS_EDQUOT", STATUS_EDQUOT)
      .value("STATUS_STOP", STATUS_STOP)
      .value("STATUS_SHUTDOWN", STATUS_SHUTDOWN)
      .value("STATUS_EOF", STATUS_EOF);
}

void ModelboxPyApiSetUpConfiguration(pybind11::module &m) {
  py::class_<modelbox::Configuration, std::shared_ptr<modelbox::Configuration>>(
      m, "Configuration", py::module_local())
      .def(py::init<>())
      .def("get_string", &modelbox::Configuration::GetString, py::arg("key"),
           py::arg("default") = py::str(""))
      .def("get_bool", &modelbox::Configuration::GetBool, py::arg("key"),
           py::arg("default") = py::bool_(false))
      .def("get_int", &modelbox::Configuration::GetInt64, py::arg("key"),
           py::arg("default") = py::int_(0))
      .def("get_float", &modelbox::Configuration::GetDouble, py::arg("key"),
           py::arg("default") = py::float_(0.0))
      .def("get_string_list", &modelbox::Configuration::GetStrings,
           py::arg("key"), py::arg("default") = py::make_tuple())
      .def("get_bool_list", &modelbox::Configuration::GetBools, py::arg("key"),
           py::arg("default") = py::make_tuple())
      .def("get_int_list", &modelbox::Configuration::GetInt64s, py::arg("key"),
           py::arg("default") = py::make_tuple())
      .def("get_float_list", &modelbox::Configuration::GetDoubles,
           py::arg("key"), py::arg("default") = py::make_tuple())
      .def("set", [](modelbox::Configuration &config, const std::string &key,
                     const py::object &obj) {
        ConfigurationSetAttributes(config, key, obj);
      });
}

py::array BufferToPyRawBuffer(modelbox::Buffer &buffer) {
  modelbox::ModelBoxDataType type = MODELBOX_TYPE_INVALID;
  buffer.Get("type", type, MODELBOX_UINT8);

  auto typesize = GetDataTypeSize(type);
  if (typesize == 0) {
    throw std::invalid_argument("buffer type is invalid");
  }

  size_t len = buffer.GetBytes() / typesize;
  const auto *const_data_ptr = buffer.ConstData();
  auto *data_ptr = const_cast<void *>(const_data_ptr);
  switch (type) {
    case MODELBOX_UINT8:
      return mkarray_via_buffer<uint8_t>(data_ptr, len);
    case MODELBOX_INT8:
      return mkarray_via_buffer<int8_t>(data_ptr, len);
    case MODELBOX_BOOL:
      return mkarray_via_buffer<bool>(data_ptr, len);
    case MODELBOX_INT16:
      return mkarray_via_buffer<int16_t>(data_ptr, len);
    case MODELBOX_UINT16:
      return mkarray_via_buffer<uint16_t>(data_ptr, len);
    case MODELBOX_INT32:
      return mkarray_via_buffer<int32_t>(data_ptr, len);
    case MODELBOX_UINT32:
      return mkarray_via_buffer<uint32_t>(data_ptr, len);
    case MODELBOX_INT64:
      return mkarray_via_buffer<int64_t>(data_ptr, len);
    case MODELBOX_UINT64:
      return mkarray_via_buffer<uint64_t>(data_ptr, len);
    case MODELBOX_FLOAT:
      return mkarray_via_buffer<float>(data_ptr, len);
    case MODELBOX_DOUBLE:
      return mkarray_via_buffer<double>(data_ptr, len);
    case MODELBOX_HALF:
      return mkarray_via_buffer<modelbox::Float16>(data_ptr, len);
    default:
      break;
  }

  return mkarray_via_buffer<uint8_t>(data_ptr, len);
}

py::array BufferToPyArrayObject(modelbox::Buffer &buffer) {
  return BufferToPyRawBuffer(buffer);
}

py::object BufferToPyString(modelbox::Buffer &buffer) {
  auto *const_data_ptr = (char *)buffer.ConstData();
  if (const_data_ptr == nullptr) {
    throw std::runtime_error("can not get buffer data.");
  }

  auto *data_ptr = const_cast<char *>(const_data_ptr);
  if (data_ptr == nullptr) {
    throw std::runtime_error("convert data to string failed.");
  }

  int len = buffer.GetBytes() / GetDataTypeSize(MODELBOX_UINT8);
  modelbox::ModelBoxDataType type = MODELBOX_TYPE_INVALID;
  buffer.Get("type", type, MODELBOX_UINT8);

  return py::cast(std::string(data_ptr, len));
}

py::object BufferToPyObject(modelbox::Buffer &buffer) {
  auto type = buffer.GetBufferType();
  if (type == modelbox::BufferEnumType::RAW) {
    return BufferToPyRawBuffer(buffer);
  }

  if (type == modelbox::BufferEnumType::IMG) {
    return BufferToPyArrayObject(buffer);
  }

  if (type == modelbox::BufferEnumType::STR) {
    return BufferToPyString(buffer);
  }

  throw std::runtime_error("invalid type");
}

void StrToBuffer(const std::shared_ptr<Buffer> &buffer,
                 const std::string &data) {
  const char *s = data.c_str();
  Py_ssize_t len = data.length();
  buffer->BuildFromHost(const_cast<char *>(s), len);
  buffer->SetGetBufferType(modelbox::BufferEnumType::STR);
}

void ListToBuffer(const std::shared_ptr<Buffer> &buffer, const py::list &data) {
  std::vector<std::vector<size_t>> vec_shapes;
  std::vector<size_t> sizes;
  std::vector<void *> source_vec;
  size_t total_bytes = 0;
  std::string info_type;
  for (const auto &item : data) {
    auto b = py::cast<py::buffer>(item);
    py::buffer_info info = b.request();
    if (info.ptr != nullptr) {
      source_vec.push_back(info.ptr);
    }

    std::vector<size_t> i_shape;
    for (auto &dim : info.shape) {
      i_shape.push_back(dim);
    }
    vec_shapes.push_back(i_shape);
    size_t bytes = Volume(i_shape) * info.itemsize;
    total_bytes += bytes;
    sizes.push_back(bytes);
    info_type = info.format;
  }

  buffer->Build(total_bytes);
  void *start = buffer->MutableData();
  int offset = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    memcpy_s((u_char *)start + offset, total_bytes, source_vec[i], sizes[i]);
    offset += sizes[i];
  }
  buffer->Set("shape", vec_shapes);
  buffer->Set("type", TypeFromFormatStr(info_type));
  buffer->SetGetBufferType(modelbox::BufferEnumType::RAW);
}

void ModelboxPyApiSetUpDevice(pybind11::module &m) {
  py::class_<modelbox::Device, std::shared_ptr<modelbox::Device>>(
      m, "Device", py::module_local())
      .def("get_device_id", &modelbox::Device::GetDeviceID)
      .def("get_type", &modelbox::Device::GetType)
      .def("get_device_desc", &modelbox::Device::GetDeviceDesc);
}

void ModelboxPyApiSetUpDataType(pybind11::handle &h) {
  py::enum_<modelbox::ModelBoxDataType>(h, "ModelBoxDataType", py::arithmetic(),
                                        py::module_local())
      .value("UINT8", MODELBOX_UINT8)
      .value("INT8", MODELBOX_INT8)
      .value("BOOL", MODELBOX_BOOL)
      .value("INT16", MODELBOX_INT16)
      .value("UINT16", MODELBOX_UINT16)
      .value("INT32", MODELBOX_INT32)
      .value("UINT32", MODELBOX_UINT32)
      .value("INT64", MODELBOX_INT64)
      .value("UINT64", MODELBOX_UINT64)
      .value("FLOAT", MODELBOX_FLOAT)
      .value("DOUBLE", MODELBOX_DOUBLE)
      .export_values();
}

py::buffer_info ModelboxPyApiSetUpBufferDefBuffer(Buffer &buffer) {
  std::vector<size_t> buffer_shape;
  auto ret = buffer.Get("shape", buffer_shape);
  if (!ret) {
    buffer_shape.push_back(buffer.GetBytes());
  }

  modelbox::ModelBoxDataType type = MODELBOX_TYPE_INVALID;
  ret = buffer.Get("type", type);
  if (!ret) {
    type = modelbox::ModelBoxDataType::MODELBOX_UINT8;
  }

  std::vector<ssize_t> shape(buffer_shape.size());
  std::vector<ssize_t> stride(buffer_shape.size());
  size_t dim_prod = 1;
  for (size_t i = 0; i < buffer_shape.size(); ++i) {
    shape[i] = buffer_shape[i];

    // We iterate over stride backwards
    stride[(buffer_shape.size() - 1) - i] =
        modelbox::GetDataTypeSize(type) * dim_prod;
    dim_prod *= buffer_shape[(buffer_shape.size() - 1) - i];
  }

  const auto *const_data_ptr = buffer.ConstData();
  auto *data_ptr = const_cast<void *>(const_data_ptr);

  return py::buffer_info(data_ptr, modelbox::GetDataTypeSize(type),
                         FormatStrFromType(type), shape.size(), shape, stride);
}

void ModelboxPyApiSetUpBuffer(pybind11::module &m) {
  using namespace pybind11::literals;  // NOLINT

  ModelboxPyApiSetUpDevice(m);

  auto h =
      py::class_<modelbox::Buffer, std::shared_ptr<modelbox::Buffer>>(
          m, "Buffer", py::module_local(), py::buffer_protocol())
          .def_buffer(ModelboxPyApiSetUpBufferDefBuffer)
          .def(py::init([](const std::shared_ptr<modelbox::Device> &device,
                           const py::buffer &b) {
                 auto buffer = std::make_shared<Buffer>(device);
                 PyBufferToBuffer(buffer, b);
                 return buffer;
               }),
               py::keep_alive<1, 2>())
          .def(py::init([](const std::shared_ptr<modelbox::Device> &device,
                           const std::string &str) {
                 auto buffer = std::make_shared<Buffer>(device);
                 StrToBuffer(buffer, str);
                 return buffer;
               }),
               py::keep_alive<1, 2>())
          .def(py::init([](const std::shared_ptr<modelbox::Device> &device,
                           const py::list &li) {
                 auto buffer = std::make_shared<Buffer>(device);
                 ListToBuffer(buffer, li);
                 return buffer;
               }),
               py::keep_alive<1, 2>())
          .def(py::init([](const std::shared_ptr<modelbox::Device> &device) {
                 auto buffer = std::make_shared<Buffer>(device);
                 return buffer;
               }),
               py::keep_alive<1, 2>())
          .def(py::init<const Buffer &>())
          .def("build", [](std::shared_ptr<Buffer> &buffer, const std::string &str) {
                 StrToBuffer(buffer, str);
               })
          .def("build", [](std::shared_ptr<Buffer> &buffer, const py::list &li) {
                 ListToBuffer(buffer, li);
               })
          .def("build", [](std::shared_ptr<Buffer> &buffer, const py::buffer &buf) {
                 PyBufferToBuffer(buffer, buf);
               })
          .def("as_bytes", [](Buffer &buffer) {
                return py::bytes{(const char *)buffer.ConstData(),
                                    buffer.GetBytes()};
               })
          .def("as_object",
               [](Buffer &buffer) -> py::object {
                 return BufferToPyObject(buffer);
               })
          .def("__str__",
               [](Buffer &buffer) {
                 return std::string((const char *)buffer.ConstData(),
                                    buffer.GetBytes());
               })
          .def("has_error", &modelbox::Buffer::HasError)
          .def("set_error", &modelbox::Buffer::SetError)
          .def("get_error_code", &modelbox::Buffer::GetErrorCode)
          .def("get_error_msg", &modelbox::Buffer::GetErrorMsg)
          .def("get_bytes", &modelbox::Buffer::GetBytes)
          .def("copy_meta",
               [](Buffer &buffer, Buffer &other) {
                 auto other_ptr = std::shared_ptr<modelbox::Buffer>(
                     &other, [](void *data) {});
                 buffer.CopyMeta(other_ptr);
               })
          .def("set",
               [](Buffer &buffer, const std::string &key,
                  py::object &obj) { BufferSetAttributes(buffer, key, obj); })
          .def("get", BufferGetAttributes);

  ModelboxPyApiSetUpDataType(h);
}

void ModelboxPyApiSetUpBufferList(pybind11::module &m) {
  using namespace pybind11::literals;  // NOLINT

  py::class_<modelbox::BufferList, std::shared_ptr<modelbox::BufferList>>(
      m, "BufferList", py::module_local())
      .def(py::init<>())
      .def(py::init<const std::shared_ptr<modelbox::Device> &>())
      .def(py::init<const std::shared_ptr<modelbox::Buffer> &>())
      .def(py::init<const std::vector<std::shared_ptr<modelbox::Buffer>> &>())
      .def("build",
           [](BufferList &bl, const std::vector<int> &shape) {
             std::vector<size_t> new_shape(shape.begin(), shape.end());
             return bl.Build(new_shape);
           })
      .def("size", &modelbox::BufferList::Size)
      .def("get_bytes", &modelbox::BufferList::GetBytes)
      .def("get_device", &modelbox::BufferList::GetDevice)
      .def(
          "push_back",
          [](BufferList &bl, Buffer &buffer) {
            auto new_buffer = std::make_shared<Buffer>(buffer);
            bl.PushBack(new_buffer);
          },
          py::keep_alive<1, 2>())
      .def(
          "push_back",
          [](BufferList &bl, const py::buffer &b) {
            auto buffer = std::make_shared<Buffer>(bl.GetDevice());
            if (PyBufferToBuffer(buffer, b) != STATUS_OK) {
              throw std::runtime_error(
                  "Failed to push back py::buffer to Buffer");
            }
            bl.PushBack(buffer);
          },
          py::keep_alive<1, 2>())
      .def("push_back",
           [](BufferList &bl, const std::string &data) {
             auto buffer = std::make_shared<Buffer>(bl.GetDevice());
             StrToBuffer(buffer, data);
             bl.PushBack(buffer);
           })
      .def("set",
           [](BufferList &bl, const std::string &key, py::object &obj) {
             for (auto &buffer : bl) {
               BufferSetAttributes(*buffer, key, obj);
             }
           })
      .def("front", &BufferList::Front)
      .def("back", &BufferList::Back)
      .def("set_error", &modelbox::BufferList::SetError)
      .def("copy_meta", &modelbox::BufferList::CopyMeta)
      .def("__len__", [](const modelbox::BufferList &bl) { return bl.Size(); })
      .def(
          "__iter__",
          [](const modelbox::BufferList &bl) {
            return py::make_iterator<
                py::return_value_policy::reference_internal>(bl.begin(),
                                                             bl.end());
          },
          py::keep_alive<0, 1>())
      .def(
          "__getitem__",
          [](modelbox::BufferList &bl, size_t i) -> std::shared_ptr<Buffer> {
            return bl.At(i);
          },
          py::keep_alive<0, 1>());
}

void ModelBoxPyApiSetUpFlowUnitEvent(pybind11::module &m) {
  py::class_<modelbox::FlowUnitError, std::shared_ptr<modelbox::FlowUnitError>>(
      m, "FlowUnitError", py::module_local())
      .def(py::init<std::string>())
      .def(py::init<std::string, std::string, modelbox::Status>())
      .def("get_description", &modelbox::FlowUnitError::GetDesc);

  py::class_<modelbox::FlowUnitEvent, std::shared_ptr<modelbox::FlowUnitEvent>>(
      m, "FlowUnitEvent", py::module_local())
      .def(py::init<>())
      .def("set_private_int",
           [](FlowUnitEvent &e, const std::string &key, long data) {
             auto private_content = std::make_shared<long>(data);
             e.SetPrivate(key, private_content);
           })
      .def("get_private_int",
           [](FlowUnitEvent &e, const std::string &key) -> long {
             auto data = e.GetPrivate(key);
             if (!data) {
               throw std::runtime_error("invalid key.");
             }

             return *((long *)(data.get()));
           })
      .def("set_private_string",
           [](FlowUnitEvent &e, const std::string &key,
              const std::string &data) {
             auto private_content = std::make_shared<std::string>(data);
             e.SetPrivate(key, private_content);
           })
      .def("get_private_string",
           [](FlowUnitEvent &e, const std::string &key) -> std::string {
             auto data = e.GetPrivate(key);
             if (!data) {
               throw std::runtime_error("invalid key.");
             }

             return *((std::string *)(data.get()));
           });
}

void ModelboxPyApiSetUpDataMeta(pybind11::module &m) {
  py::class_<modelbox::DataMeta, std::shared_ptr<modelbox::DataMeta>>(
      m, "DataMeta", py::module_local())
      .def(py::init<>())
      .def("set_private_int",
           [](DataMeta &e, const std::string &key, long data) {
             auto private_content = std::make_shared<long>(data);
             e.SetMeta(key, private_content);
           })
      .def("get_private_int",
           [](DataMeta &e, const std::string &key) -> long {
             auto data = e.GetMeta(key);
             if (!data) {
               throw std::runtime_error("invalid key.");
             }

             return *((long *)(data.get()));
           })
      .def("set_private_string",
           [](DataMeta &e, const std::string &key, const std::string &data) {
             auto private_content = std::make_shared<std::string>(data);
             e.SetMeta(key, private_content);
           })
      .def("get_private_string",
           [](DataMeta &e, const std::string &key) -> std::string {
             auto data = e.GetMeta(key);
             if (!data) {
               throw std::runtime_error("invalid key.");
             }

             return *((std::string *)(data.get()));
           });
}

void ModelboxPyApiSetUpSessionContext(pybind11::module &m) {
  py::class_<modelbox::SessionContext,
             std::shared_ptr<modelbox::SessionContext>>(m, "SessionContext",
                                                        py::module_local())
      .def(py::init<>())
      .def("set_private_int",
           [](SessionContext &ctx, const std::string &key, long data) {
             auto private_content = std::make_shared<long>(data);
             ctx.SetPrivate(key, private_content);
           })
      .def("get_private_int",
           [](SessionContext &ctx, const std::string &key) -> long {
             auto data = ctx.GetPrivate(key);
             if (!data) {
               throw std::runtime_error("invalid key.");
             }

             return *((long *)(data.get()));
           })
      .def("set_private_string",
           [](SessionContext &ctx, const std::string &key,
              const std::string &data) {
             auto private_content = std::make_shared<std::string>(data);
             ctx.SetPrivate(key, private_content);
           })
      .def("get_private_string",
           [](SessionContext &ctx, const std::string &key) -> std::string {
             auto data = ctx.GetPrivate(key);
             if (!data) {
               throw std::runtime_error("invalid key.");
             }

             return *((std::string *)(data.get()));
           })
      .def("set_private",
           [](SessionContext &ctx, const std::string &key, py::object &obj) {
             SessionContextSetAttributes(ctx, key, obj);
           })
      .def("get_private",
           [](SessionContext &ctx, const std::string &key) {
             return SessionContextGetAttributes(ctx, key);
           })
      .def("get_session_config", &modelbox::SessionContext::GetConfig)
      .def("get_session_id", &modelbox::SessionContext::GetSessionId);
}

void ModelboxPyApiSetUpDataContext(pybind11::module &m) {
  py::class_<modelbox::ExternalData, std::shared_ptr<modelbox::ExternalData>>(
      m, "ExternalData", py::module_local())
      .def("create_buffer_list", &modelbox::ExternalData::CreateBufferList)
      .def("send", &modelbox::ExternalData::Send)
      .def("get_session_context", &modelbox::ExternalData::GetSessionContext)
      .def("get_session_config", &modelbox::ExternalData::GetSessionConfig)
      .def("close", &modelbox::ExternalData::Close);

  py::class_<modelbox::DataContext, std::shared_ptr<modelbox::DataContext>>(
      m, "DataContext", py::module_local())
      .def("input", static_cast<std::shared_ptr<modelbox::BufferList> (
                        modelbox::DataContext::*)(const std::string &) const>(
                        &modelbox::DataContext::Input))
      .def("output", static_cast<std::shared_ptr<modelbox::BufferList> (
                         modelbox::DataContext::*)(const std::string &)>(
                         &modelbox::DataContext::Output))
      .def("external", &modelbox::DataContext::External)
      .def("event", &modelbox::DataContext::Event)
      .def("has_error", &modelbox::DataContext::HasError)
      .def("send_event", &modelbox::DataContext::SendEvent)
      .def("set_private_string",
           [](DataContext &ctx, const std::string &key,
              const std::string &data) {
             auto private_content = std::make_shared<std::string>(data);
             ctx.SetPrivate(key, private_content);
           })
      .def("set_private_int",
           [](DataContext &ctx, const std::string &key, long data) {
             auto private_content = std::make_shared<long>(data);
             ctx.SetPrivate(key, private_content);
           })
      .def("get_private_string",
           [](DataContext &ctx, const std::string &key) -> std::string {
             auto data = ctx.GetPrivate(key);
             if (!data) {
               throw std::runtime_error("invalid key.");
             }

             return *((std::string *)(data.get()));
           })
      .def("get_private_int",
           [](DataContext &ctx, const std::string &key) -> long {
             auto data = ctx.GetPrivate(key);
             if (!data) {
               throw std::runtime_error("invalid key.");
             }

             return *((long *)(data.get()));
           })
      .def("set_private",
           [](DataContext &ctx, const std::string &key, py::object &obj) {
             DataContextSetAttributes(ctx, key, obj);
           })
      .def("get_private",
           [](DataContext &ctx, const std::string &key) {
             return DataContextGetAttributes(ctx, key);
           })
      .def("get_input_meta", &modelbox::DataContext::GetInputMeta)
      .def("get_input_group_meta", &modelbox::DataContext::GetInputGroupMeta)
      .def("set_output_meta", &modelbox::DataContext::SetOutputMeta)
      .def("get_session_config", &modelbox::DataContext::GetSessionConfig)
      .def("get_session_context", &modelbox::DataContext::GetSessionContext);
}

void ModelboxPyApiSetUpGeneric(pybind11::module &m) {
  ModelBoxPyApiSetUpFlowUnitEvent(m);
  ModelboxPyApiSetUpDataMeta(m);
  ModelboxPyApiSetUpSessionContext(m);
  ModelboxPyApiSetUpDataContext(m);
}

class PyFlowUnit : public modelbox::FlowUnit {
 public:
  using FlowUnit::FlowUnit;  // Inherit constructors
  Status Open(const std::shared_ptr<Configuration> &copnfigure) override {
    PYBIND11_OVERLOAD_PURE(Status, FlowUnit, Open, copnfigure);
  }

  Status Close() override { PYBIND11_OVERLOAD_PURE(Status, FlowUnit, Close, ); }

  Status Process(std::shared_ptr<DataContext> data_ctx) override {
    PYBIND11_OVERLOAD_PURE(Status, IFlowUnit, Process, data_ctx);
  }

  Status DataPre(std::shared_ptr<DataContext> data_ctx) override {
    PYBIND11_OVERLOAD_PURE(Status, IFlowUnit, DataPre, data_ctx);
  }

  Status DataPost(std::shared_ptr<DataContext> data_ctx) override {
    PYBIND11_OVERLOAD_PURE(Status, IFlowUnit, DataPost, data_ctx);
  }

  Status DataGroupPre(std::shared_ptr<DataContext> data_ctx) override {
    PYBIND11_OVERLOAD_PURE(Status, IFlowUnit, DataGroupPre, data_ctx);
  }

  Status DataGroupPost(std::shared_ptr<DataContext> data_ctx) override {
    PYBIND11_OVERLOAD_PURE(Status, IFlowUnit, DataGroupPost, data_ctx);
  }

  std::shared_ptr<Device> GetBindDevice() {
    PYBIND11_OVERLOAD(std::shared_ptr<Device>, FlowUnit, GetBindDevice, );
  }

  std::shared_ptr<ExternalData> CreateExternalData() const {
    PYBIND11_OVERLOAD(std::shared_ptr<ExternalData>, FlowUnit,
                      CreateExternalData);
  }
};

void ModelboxPyApiSetUpFlowUnit(pybind11::module &m) {
  py::class_<modelbox::FlowUnit, PyFlowUnit>(m, "FlowUnit", py::module_local())
      .def(py::init<>())
      .def("open", &modelbox::FlowUnit::Open)
      .def("close", &modelbox::FlowUnit::Close)
      .def("process", &modelbox::FlowUnit::Process)
      .def("data_pre", &modelbox::FlowUnit::DataPre)
      .def("data_post", &modelbox::FlowUnit::DataPost)
      .def("data_group_pre", &modelbox::FlowUnit::DataGroupPre)
      .def("data_group_post", &modelbox::FlowUnit::DataGroupPost)
      .def("get_bind_device", &modelbox::FlowUnit::GetBindDevice)
      .def("create_external_data", &modelbox::FlowUnit::CreateExternalData)
      .def(
          "create_buffer",
          [](modelbox::FlowUnit &flow,
             py::buffer &b) -> std::shared_ptr<Buffer> {
            auto buffer = std::make_shared<Buffer>(flow.GetBindDevice());
            if (PyBufferToBuffer(buffer, b) != STATUS_OK) {
              throw std::runtime_error("create buffer failed.");
            }
            return buffer;
          },
          py::keep_alive<0, 1>());
}

void ModelboxPyApiSetUpEngine(pybind11::module &m) {
  py::class_<modelbox::ModelBoxEngine,
             std::shared_ptr<modelbox::ModelBoxEngine>>(m, "ModelBoxEngine")
      .def(py::init<>())
      .def(
          "init",
          [](ModelBoxEngine &env,
             std::shared_ptr<modelbox::Configuration> &config) {
            return env.Init(config);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "init",
          [](ModelBoxEngine &env,
             std::unordered_map<std::string, std::string> &config) {
            auto configuration = std::make_shared<modelbox::Configuration>();
            for (auto &iter : config) {
              configuration->SetProperty(iter.first, iter.second);
            }
            return env.Init(configuration);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("shutdown", &modelbox::ModelBoxEngine::ShutDown,
           py::call_guard<py::gil_scoped_release>())
      .def("close", &modelbox::ModelBoxEngine::Close,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "create_input",
          [](ModelBoxEngine &env, const std::set<std::string> &port_map) {
            py::gil_scoped_release release;
            return env.CreateInput(port_map);
          },
          py::keep_alive<0, 1>())
      .def(
          "execute",
          [](ModelBoxEngine &env, const std::string &name,
             std::map<std::string, std::string> &config,
             std::map<std::string, std::shared_ptr<DataHandler>> &data) {
            py::gil_scoped_release release;
            return env.Execute(name, config, data);
          },
          py::keep_alive<0, 1>())
      .def(
          "execute",
          [](ModelBoxEngine &env, const std::string &name,
             std::map<std::string, std::string> &config,
             std::shared_ptr<DataHandler> &data) {
            py::gil_scoped_release release;
            return env.Execute(name, config, data);
          },
          py::keep_alive<0, 1>());
}

void ModelboxPyApiSetUpDataHandler(pybind11::module &m) {
  py::class_<modelbox::DataHandler, std::shared_ptr<modelbox::DataHandler>>(
      m, "DataHandler")
      .def(py::init<>())
      .def("close", &modelbox::DataHandler::Close,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "__iter__", [](DataHandler &data) -> DataHandler & { return data; },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "__next__",
          [](DataHandler &data) {
            py::gil_scoped_release release;
            auto buffer = data.GetData();
            if (buffer == nullptr) {
              throw pybind11::stop_iteration();
            }
            return buffer;
          },
          py::keep_alive<0, 1>())
      .def(
          "__getitem__",
          [](DataHandler &data, const std::string &key) {
            auto sub_data = data.GetDataHandler(key);
            if (sub_data == nullptr) {
              throw pybind11::index_error();
            }
            return sub_data;
          },
          py::call_guard<py::gil_scoped_release>())
      .def("setmeta",
           static_cast<modelbox::Status (modelbox::DataHandler::*)(
               const std::string &, const std::string &)>(
               &modelbox::DataHandler::SetMeta),
           py::call_guard<py::gil_scoped_release>())
      .def("pushdata",
           static_cast<modelbox::Status (modelbox::DataHandler::*)(
               std::shared_ptr<modelbox::Buffer> &, const std::string &)>(
               &modelbox::DataHandler::PushData),
           py::call_guard<py::gil_scoped_release>())
      .def("pushdata",
           static_cast<modelbox::Status (modelbox::DataHandler::*)(
               std::shared_ptr<DataHandler> &, const std::string &)>(
               &modelbox::DataHandler::PushData),
           py::call_guard<py::gil_scoped_release>())
      .def("get_datahandler", &modelbox::DataHandler::GetDataHandler,
           py::call_guard<py::gil_scoped_release>())
      .def("set_datahandler", &modelbox::DataHandler::SetDataHandler,
           py::call_guard<py::gil_scoped_release>());
}

void ModelboxPyApiSetUpFlowGraphDesc(pybind11::module &m) {
  py::class_<modelbox::FlowGraphDesc, std::shared_ptr<modelbox::FlowGraphDesc>>(
      m, "FlowGraphDesc")
      .def(py::init<>())
      .def("set_queue_size", &modelbox::FlowGraphDesc::SetQueueSize)
      .def("set_batch_size", &modelbox::FlowGraphDesc::SetBatchSize)
      .def("set_drivers_dir", &modelbox::FlowGraphDesc::SetDriversDir)
      .def("set_skip_default_drivers",
           &modelbox::FlowGraphDesc::SetSkipDefaultDrivers)
      .def("set_profile_dir", &modelbox::FlowGraphDesc::SetProfileDir)
      .def("set_profile_trace_enable",
           &modelbox::FlowGraphDesc::SetProfileTraceEnable)
      .def("add_input", &modelbox::FlowGraphDesc::AddInput,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "add_output",
          [](FlowGraphDesc &self, const std::string &output_name,
             const std::shared_ptr<FlowPortDesc> &source_node_port) {
            return self.AddOutput(output_name, source_node_port);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "add_output",
          [](FlowGraphDesc &self, const std::string &output_name,
             const std::shared_ptr<FlowNodeDesc> &source_node) {
            return self.AddOutput(output_name, source_node);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "add_node",
          [](FlowGraphDesc &self, const std::string &flowunit_name,
             const std::string &device, const std::vector<std::string> &config,
             const std::unordered_map<std::string,
                                      std::shared_ptr<FlowPortDesc>>
                 &source_node_ports) {
            return self.AddNode(flowunit_name, device, config,
                                source_node_ports);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "add_node",
          [](FlowGraphDesc &self, const std::string &flowunit_name,
             const std::string &device, const std::vector<std::string> &config,
             const std::shared_ptr<FlowNodeDesc> &source_node) {
            return self.AddNode(flowunit_name, device, config, source_node);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "add_node",
          [](FlowGraphDesc &self, const std::string &flowunit_name,
             const std::string &device,
             const std::unordered_map<std::string,
                                      std::shared_ptr<FlowPortDesc>>
                 &source_node_ports) {
            return self.AddNode(flowunit_name, device, source_node_ports);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "add_node",
          [](FlowGraphDesc &self, const std::string &flowunit_name,
             const std::string &device,
             const std::shared_ptr<FlowNodeDesc> &source_node) {
            return self.AddNode(flowunit_name, device, source_node);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "add_node",
          [](FlowGraphDesc &self, const std::string &flowunit_name,
             const std::string &device,
             const std::vector<std::string> &config) {
            return self.AddNode(flowunit_name, device, config);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "add_function",
          [](FlowGraphDesc &self,
             const std::function<StatusCode(std::shared_ptr<DataContext>)>
                 &func,
             const std::vector<std::string> &input_name_list,
             const std::vector<std::string> &output_name_list,
             const std::unordered_map<std::string,
                                      std::shared_ptr<FlowPortDesc>>
                 &source_node_ports) {
            return self.AddFunction(func, input_name_list, output_name_list,
                                    source_node_ports);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "add_function",
          [](FlowGraphDesc &self,
             const std::function<StatusCode(std::shared_ptr<DataContext>)>
                 &func,
             const std::vector<std::string> &input_name_list,
             const std::vector<std::string> &output_name_list,
             const std::shared_ptr<FlowNodeDesc> &source_node) {
            return self.AddFunction(func, input_name_list, output_name_list,
                                    source_node);
          },
          py::call_guard<py::gil_scoped_release>());
}

void ModelboxPyApiSetUpFlowNodeDesc(pybind11::module &m) {
  py::class_<modelbox::FlowNodeDesc, std::shared_ptr<modelbox::FlowNodeDesc>>(
      m, "FlowNodeDesc")
      .def(py::init<const std::string &>())
      .def("set_node_name", &modelbox::FlowNodeDesc::SetNodeName)
      .def("get_node_name", &modelbox::FlowNodeDesc::GetNodeName)
      .def("__getitem__",
           [](modelbox::FlowNodeDesc &node_desc, const std::string &output_name)
               -> std::shared_ptr<modelbox::FlowPortDesc> {
             return node_desc[output_name];
           })
      .def("__getitem__",
           [](modelbox::FlowNodeDesc &node_desc,
              size_t port_idx) -> std::shared_ptr<modelbox::FlowPortDesc> {
             return node_desc[port_idx];
           });
}

void ModelboxPyApiSetUpFlowPortDesc(pybind11::module &m) {
  py::class_<modelbox::FlowPortDesc, std::shared_ptr<modelbox::FlowPortDesc>>(
      m, "FlowPortDesc")
      .def(py::init<std::shared_ptr<FlowNodeDesc>, const std::string &>())
      .def("get_node_name", &modelbox::FlowPortDesc::GetNodeName)
      .def("get_port_name", &modelbox::FlowPortDesc::GetPortName);
}

void ModelboxPyApiSetUpFlowStreamIO(pybind11::module &m) {
  py::class_<PythonFlowStreamIO, std::shared_ptr<PythonFlowStreamIO>>(
      m, "FlowStreamIO")
      .def(py::init<std::shared_ptr<FlowStreamIO>>())
      .def("create_buffer", &PythonFlowStreamIO::CreateBuffer,
           py::keep_alive<0, 1>(), py::call_guard<py::gil_scoped_release>())
      .def(
          "create_buffer",
          [](PythonFlowStreamIO &self, const py::buffer &data) {
            auto buffer = self.CreateBuffer();
            if (PyBufferToBuffer(buffer, data) != STATUS_OK) {
              throw std::runtime_error("Failed to create buffer");
            }
            return buffer;
          },
          py::keep_alive<0, 1>())
      .def(
          "create_buffer",
          [](PythonFlowStreamIO &self, const std::string &data) {
            auto buffer = self.CreateBuffer();
            StrToBuffer(buffer, data);
            return buffer;
          },
          py::keep_alive<0, 1>(), py::call_guard<py::gil_scoped_release>())
      .def("send", &PythonFlowStreamIO::Send,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "send",
          [](PythonFlowStreamIO &self, const std::string &input_name,
             const py::buffer &data) {
            auto buffer = self.CreateBuffer();
            {
              py::gil_scoped_acquire ac;
              if (PyBufferToBuffer(buffer, data) != STATUS_OK) {
                throw std::runtime_error("Failed to send buffer");
              }
            }
            return self.Send(input_name, buffer);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "send",
          [](PythonFlowStreamIO &self, const std::string &input_name,
             const std::string &data) {
            auto buffer = self.CreateBuffer();
            StrToBuffer(buffer, data);
            return self.Send(input_name, buffer);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "recv",
          [](PythonFlowStreamIO &self, const std::string &output_name,
             std::shared_ptr<Buffer> &buffer,
             size_t timeout) -> modelbox::Status {
            std::shared_ptr<Buffer> out_buffer;
            auto ret = self.Recv(output_name, out_buffer, timeout);
            if (ret != STATUS_OK) {
              return ret;
            }

            *buffer = *out_buffer;
            return modelbox::STATUS_OK;
          },
          py::arg("output_name"), py::arg("buffer"), py::arg("timeout") = 0,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "recv",
          [](PythonFlowStreamIO &self, const std::string &output_name,
             size_t timeout) {
            std::shared_ptr<Buffer> buffer;
            self.Recv(output_name, buffer, timeout);
            return buffer;
          },
          py::keep_alive<0, 1>(), py::call_guard<py::gil_scoped_release>())
      .def(
          "recv",
          [](PythonFlowStreamIO &self, const std::string &output_name) {
            std::shared_ptr<Buffer> buffer;
            self.Recv(output_name, buffer, 0);
            return buffer;
          },
          py::keep_alive<0, 1>(), py::call_guard<py::gil_scoped_release>())
      .def("close_input", &PythonFlowStreamIO::CloseInput,
           py::call_guard<py::gil_scoped_release>());
}

void ModelBoxPyApiSetUpExternalDataMapSimple(pybind11::module &m) {
  py::class_<modelbox::ExternalDataSimple,
             std::shared_ptr<modelbox::ExternalDataSimple>>(
      m, "ExternalDataSimple")
      .def(py::init<std::shared_ptr<ExternalDataMap> &>())
      .def("pushdata",
           [](ExternalDataSimple &extern_data_simple,
              const std::string &port_name,
              std::shared_ptr<BufferList> &bufferlist) {
             py::gil_scoped_release release;
             return extern_data_simple.PushData(port_name, bufferlist);
           })
      .def("getresult",
           [](ExternalDataSimple &extern_data_simple,
              const std::string &port_name,
              int timeout = 0) -> std::shared_ptr<Buffer> {
             py::gil_scoped_release release;
             std::shared_ptr<Buffer> buffer;
             if (STATUS_OK ==
                 extern_data_simple.GetResult(port_name, buffer, timeout)) {
               return buffer;
             }
             return nullptr;
           });
}

void ModelboxPyApiSetUpModel(pybind11::module &m) {
  py::class_<PythonModel, std::shared_ptr<PythonModel>>(m, "Model",
                                                        py::module_local())
      .def(py::init<std::string, std::string, size_t, std::string,
                    std::string>())
      .def("add_path", &PythonModel::AddPath,
           py::call_guard<py::gil_scoped_release>())
      .def("start", &PythonModel::Start,
           py::call_guard<py::gil_scoped_release>())
      .def("stop", &PythonModel::Stop, py::call_guard<py::gil_scoped_release>())
      .def("infer", &PythonModel::Infer,
           py::call_guard<py::gil_scoped_release>())
      .def("infer_batch", &PythonModel::InferBatch,
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace modelbox
