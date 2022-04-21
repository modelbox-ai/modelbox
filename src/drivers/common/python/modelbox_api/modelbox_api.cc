
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
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <securec.h>

#include <string>
#include "modelbox/data_context.h"
#include "modelbox/external_data_simple.h"
#include "modelbox/flow.h"
#include "modelbox/modelbox_engine.h"
#include "modelbox/type.h"
#include "python_log.h"

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

  static constexpr auto name() { return _("float16"); }
};

namespace modelbox {

static std::unordered_map<int, std::string> kTypeToNPType;
static std::unordered_map<std::string, ModelBoxDataType> kNPTypeToType;

void BuildTypeToNumpyType() {
  if (kTypeToNPType.empty() && kNPTypeToType.empty()) {
    kTypeToNPType[MODELBOX_UINT8] = py::format_descriptor<uint8_t>::format();
    kTypeToNPType[MODELBOX_INT8] = py::format_descriptor<int8_t>::format();
    kTypeToNPType[MODELBOX_BOOL] = py::format_descriptor<bool>::format();
    kTypeToNPType[MODELBOX_INT16] = py::format_descriptor<int16_t>::format();
    kTypeToNPType[MODELBOX_UINT16] = py::format_descriptor<uint16_t>::format();
    kTypeToNPType[MODELBOX_INT32] = py::format_descriptor<int32_t>::format();
    kTypeToNPType[MODELBOX_UINT32] = py::format_descriptor<uint32_t>::format();
    kTypeToNPType[MODELBOX_INT64] = py::format_descriptor<int64_t>::format();
    kTypeToNPType[MODELBOX_UINT64] = py::format_descriptor<uint64_t>::format();
    kTypeToNPType[MODELBOX_FLOAT] = py::format_descriptor<float>::format();
    kTypeToNPType[MODELBOX_DOUBLE] = py::format_descriptor<double>::format();
    kNPTypeToType[py::format_descriptor<uint8_t>::format()] = MODELBOX_UINT8;
    kNPTypeToType[py::format_descriptor<int8_t>::format()] = MODELBOX_INT8;
    kNPTypeToType[py::format_descriptor<bool>::format()] = MODELBOX_BOOL;
    kNPTypeToType[py::format_descriptor<int16_t>::format()] = MODELBOX_INT16;
    kNPTypeToType[py::format_descriptor<uint16_t>::format()] = MODELBOX_UINT16;
    kNPTypeToType[py::format_descriptor<int32_t>::format()] = MODELBOX_INT32;
    kNPTypeToType[py::format_descriptor<uint32_t>::format()] = MODELBOX_UINT32;
    kNPTypeToType[py::format_descriptor<int64_t>::format()] = MODELBOX_INT64;
    kNPTypeToType[py::format_descriptor<uint64_t>::format()] = MODELBOX_UINT64;
    kNPTypeToType[py::format_descriptor<float>::format()] = MODELBOX_FLOAT;
    kNPTypeToType[py::format_descriptor<double>::format()] = MODELBOX_DOUBLE;
    kNPTypeToType["e"] = MODELBOX_FLOAT;
    kNPTypeToType["l"] = MODELBOX_INT64;
  }

  return;
}

std::string FormatStrFromType(const modelbox::ModelBoxDataType &type) {
  BuildTypeToNumpyType();

  auto iter = kTypeToNPType.find(type);
  if (iter == kTypeToNPType.end()) {
    std::string errmsg = "invalid modelbox data type: ";
    errmsg += std::to_string(type);
    throw std::runtime_error(errmsg);
  }

  return iter->second;
}

modelbox::ModelBoxDataType TypeFromFormatStr(const std::string &format) {
  BuildTypeToNumpyType();

  auto iter = kNPTypeToType.find(format);
  if (iter == kNPTypeToType.end()) {
    throw std::runtime_error("invalid numpy data type: " + format);
  }

  return iter->second;
}

typedef bool (*pBufferTypeChangeFunc)(Buffer &, const std::string &, py::object,
                                      py::object);

template <typename PyType, typename FlowType>
bool BufferSet(Buffer &buffer, const std::string &key, py::object set_obj,
               py::object cast_obj) {
  if (py::isinstance<PyType>(set_obj)) {
    buffer.Set(key, cast_obj.cast<FlowType>());
    return true;
  }

  return false;
}

static pBufferTypeChangeFunc kChangeFunc[] = {
    BufferSet<py::float_, double>,
    BufferSet<ModelBoxDataType, ModelBoxDataType>,
    BufferSet<py::str, std::string>, BufferSet<py::bool_, bool>,
    BufferSet<py::int_, long>};

static pBufferTypeChangeFunc kListChangeFunc[] = {
    BufferSet<py::float_, std::vector<double>>,
    BufferSet<py::str, std::vector<std::string>>,
    BufferSet<py::bool_, std::vector<bool>>,
    BufferSet<py::int_, std::vector<long>>};

void PythonBufferSet(Buffer &buffer, const std::string &key, py::object &obj) {
  for (auto &pfunc : kChangeFunc) {
    if (pfunc(buffer, key, obj, obj)) {
      return;
    }
  }

  if (py::isinstance<py::list>(obj)) {
    py::list obj_list = obj.cast<py::list>();
    for (auto &pfunc : kListChangeFunc) {
      if (pfunc(buffer, key, obj_list[0], obj_list)) {
        return;
      }
    }

    buffer.Set(key, obj_list);
    return;
  }

  throw std::invalid_argument("invalid data type " +
                              py::str(obj).cast<std::string>() + " for key " +
                              key);
}

void ModelboxPyApiSetUpLog(pybind11::module &m) {
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
          .def(py::self == py::self)
          .def(py::self == StatusCode())
          .def(py::self != py::self)
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

typedef bool (*pConfigSetFunc)(Configuration &config, const std::string &key,
                               py::object set_obj, py::object cast_obj);

template <typename PyType, typename FlowType>
bool ConfigSet(Configuration &config, const std::string &key,
               py::object set_obj, py::object cast_obj) {
  if (py::isinstance<PyType>(set_obj)) {
    config.SetProperty(key, cast_obj.cast<FlowType>());
    return true;
  }

  return false;
}

static pConfigSetFunc kConfigFunc[] = {
    ConfigSet<py::float_, double>, ConfigSet<py::str, std::string>,
    ConfigSet<py::bool_, bool>, ConfigSet<py::int_, long>};

static pConfigSetFunc kConfigListFunc[] = {
    ConfigSet<py::float_, std::vector<double>>,
    ConfigSet<py::str, std::vector<std::string>>,
    ConfigSet<py::bool_, std::vector<bool>>,
    ConfigSet<py::int_, std::vector<long>>};

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
                     py::object obj) {
        for (auto &pfunc : kConfigFunc) {
          if (pfunc(config, key, obj, obj)) {
            return;
          }
        }

        if (py::isinstance<py::list>(obj)) {
          py::list obj_list = obj.cast<py::list>();
          for (auto &pfunc : kConfigListFunc) {
            if (pfunc(config, key, obj_list[0], obj_list)) {
              return;
            }
          }
        }

        throw std::invalid_argument("invalid data type " +
                                    py::str(obj).cast<std::string>() +
                                    " for key " + key);
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
  auto const_data_ptr = buffer.ConstData();
  auto data_ptr = const_cast<void *>(const_data_ptr);
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
  auto const_data_ptr = (char *)buffer.ConstData();
  if (const_data_ptr == nullptr) {
    throw std::runtime_error("can not get buffer data.");
  }

  char *data_ptr = const_cast<char *>(const_data_ptr);
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
  } else if (type == modelbox::BufferEnumType::IMG) {
    return BufferToPyArrayObject(buffer);
  } else if (type == modelbox::BufferEnumType::STR) {
    return BufferToPyString(buffer);
  }

  throw std::runtime_error("invalid type");
}

typedef bool (*pInterTypeToPyTypeFunc)(std::size_t hash_code, Any *value,
                                       py::object &ret);

template <typename InterTyper, typename PyType>
bool InterTypeToPyType(std::size_t hash_code, Any *value, py::object &ret) {
  if (typeid(InterTyper).hash_code() == hash_code) {
    auto *data = any_cast<InterTyper>(value);
    if (data == nullptr) {
      return false;
    }
    ret = py::cast(*data);
    return true;
  }

  return false;
}

template <typename InterTyper, typename PyType>
bool InterTypeToPyListType(std::size_t hash_code, Any *value, py::object &ret) {
  if (typeid(std::vector<InterTyper>).hash_code() == hash_code) {
    auto *data = any_cast<std::vector<InterTyper>>(value);
    if (data == nullptr) {
      return false;
    }

    py::list li;
    for_each(data->begin(), data->end(),
             [&li](const InterTyper &val) { li.append(val); });
    ret = li;
    return true;
  }

  return false;
}

template <typename InterTyper, typename PyType>
bool InterTypeToPyListInListType(std::size_t hash_code, Any *value,
                                 py::object &ret) {
  if (typeid(std::vector<std::vector<InterTyper>>).hash_code() == hash_code) {
    auto *data = any_cast<std::vector<std::vector<InterTyper>>>(value);
    if (data == nullptr) {
      MBLOG_ERROR << "data is nullptr.";
      return false;
    }

    py::list li;
    for_each(data->begin(), data->end(),
             [&li](const std::vector<InterTyper> &val) {
               py::list item_li;
               for (const auto &item : val) {
                 item_li.append(item);
               }
               li.append(item_li);
             });
    ret = li;
    return true;
  }

  return false;
}

static pInterTypeToPyTypeFunc kTypeFunc[] = {
    InterTypeToPyType<ModelBoxDataType, ModelBoxDataType>,
    InterTypeToPyType<int, py::int_>,
    InterTypeToPyType<unsigned int, py::int_>,
    InterTypeToPyType<long, py::int_>,
    InterTypeToPyType<unsigned long, py::int_>,
    InterTypeToPyType<char, py::int_>,
    InterTypeToPyType<unsigned char, py::int_>,
    InterTypeToPyType<float, py::float_>,
    InterTypeToPyType<double, py::float_>,
    InterTypeToPyType<std::string, py::str>,
    InterTypeToPyType<bool, py::bool_>};

static pInterTypeToPyTypeFunc kListTypeFunc[] = {
    InterTypeToPyListType<int, py::list>,
    InterTypeToPyListType<unsigned int, py::list>,
    InterTypeToPyListType<long, py::list>,
    InterTypeToPyListType<unsigned long, py::list>,
    InterTypeToPyListType<char, py::list>,
    InterTypeToPyListType<unsigned char, py::list>,
    InterTypeToPyListType<float, py::list>,
    InterTypeToPyListType<double, py::list>,
    InterTypeToPyListType<std::string, py::list>,
    InterTypeToPyListType<bool, py::list>};

static pInterTypeToPyTypeFunc kListInListTypeFunc[] = {
    InterTypeToPyListInListType<float, py::list>,
    InterTypeToPyListInListType<double, py::list>,
    InterTypeToPyListInListType<int, py::list>,
    InterTypeToPyListInListType<unsigned int, py::list>,
    InterTypeToPyListInListType<long, py::list>,
    InterTypeToPyListInListType<unsigned long, py::list>,
};

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
    throw std::runtime_error("can not get buffer shape.");
  }

  modelbox::ModelBoxDataType type = MODELBOX_TYPE_INVALID;
  ret = buffer.Get("type", type);
  if (!ret) {
    throw std::runtime_error("can not get buffer type.");
  }

  std::vector<ssize_t> shape(buffer_shape.size()), stride(buffer_shape.size());
  size_t dim_prod = 1;
  for (size_t i = 0; i < buffer_shape.size(); ++i) {
    shape[i] = buffer_shape[i];

    // We iterate over stride backwards
    stride[(buffer_shape.size() - 1) - i] =
        modelbox::GetDataTypeSize(type) * dim_prod;
    dim_prod *= buffer_shape[(buffer_shape.size() - 1) - i];
  }

  auto const_data_ptr = buffer.ConstData();
  auto data_ptr = const_cast<void *>(const_data_ptr);

  return py::buffer_info(data_ptr, modelbox::GetDataTypeSize(type),
                         FormatStrFromType(type), shape.size(), shape, stride);
}

py::object ModelboxPyApiSetUpBufferDefGet(Buffer &buffer,
                                          const std::string &key) {
  auto ret = buffer.Get(key);
  if (!std::get<1>(ret)) {
    throw std::invalid_argument("can not find buffer meta: " + key);
  }

  auto *value = std::get<0>(ret);
  auto hash_code = value->type().hash_code();
  py::object ret_data;
  for (auto &pfunc : kTypeFunc) {
    if (pfunc(hash_code, value, ret_data)) {
      return ret_data;
    }
  }

  for (auto &pfunc : kListTypeFunc) {
    if (pfunc(hash_code, value, ret_data)) {
      return ret_data;
    }
  }

  for (auto &pfunc : kListInListTypeFunc) {
    if (pfunc(hash_code, value, ret_data)) {
      return ret_data;
    }
  }

  throw std::invalid_argument("invalid data type " +
                              std::string(value->type().name()) +
                              " for buffer meta " + key);
}

void ModelboxPyApiSetUpBuffer(pybind11::module &m) {
  using namespace pybind11::literals;

  ModelboxPyApiSetUpDevice(m);

  auto h = py::class_<modelbox::Buffer, std::shared_ptr<modelbox::Buffer>>(
               m, "Buffer", py::module_local(), py::buffer_protocol())
               .def_buffer(ModelboxPyApiSetUpBufferDefBuffer)
               .def(py::init([](std::shared_ptr<modelbox::Device> device,
                                py::buffer b) {
                      py::buffer_info info = b.request();
                      std::vector<size_t> i_shape;
                      for (auto &dim : info.shape) {
                        i_shape.push_back(dim);
                      }

                      if (info.shape.size() == 0) {
                        throw std::runtime_error("can not accept empty numpy.");
                      }

                      size_t bytes = Volume(i_shape) * info.itemsize;
                      auto buffer = std::make_shared<Buffer>(device);
                      buffer->BuildFromHost(info.ptr, bytes);
                      buffer->Set("shape", i_shape);
                      buffer->Set("type", TypeFromFormatStr(info.format));
                      buffer->SetGetBufferType(modelbox::BufferEnumType::RAW);
                      return buffer;
                    }),
                    py::keep_alive<1, 2>())
               .def(py::init([](std::shared_ptr<modelbox::Device> device,
                                const std::string &str) {
                      auto buffer = std::make_shared<Buffer>(device);
                      const char *s = str.c_str();
                      Py_ssize_t len = str.length();
                      buffer->BuildFromHost(const_cast<char *>(s), len);
                      buffer->SetGetBufferType(modelbox::BufferEnumType::STR);
                      return buffer;
                    }),
                    py::keep_alive<1, 2>())
               .def(py::init([](std::shared_ptr<modelbox::Device> device,
                                py::list li) {
                      auto buffer = std::make_shared<Buffer>(device);
                      std::vector<std::vector<size_t>> vec_shapes;
                      std::vector<size_t> sizes;
                      std::vector<void *> source_vec;
                      size_t total_bytes = 0;
                      std::string info_type;
                      for (auto &item : li) {
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
                        memcpy_s((u_char *)start + offset, total_bytes,
                                 source_vec[i], sizes[i]);
                        offset += sizes[i];
                      }
                      buffer->Set("shape", vec_shapes);
                      buffer->Set("type", TypeFromFormatStr(info_type));
                      buffer->SetGetBufferType(modelbox::BufferEnumType::RAW);
                      return buffer;
                    }),
                    py::keep_alive<1, 2>())
               .def(py::init<const Buffer &>())
               .def("as_object",
                    [](Buffer &buffer) -> py::object {
                      return BufferToPyObject(buffer);
                    })
               .def("has_error", &modelbox::Buffer::HasError)
               .def("set_error", &modelbox::Buffer::SetError)
               .def("get_error", &modelbox::Buffer::GetError)
               .def("get_bytes", &modelbox::Buffer::GetBytes)
               .def("copy_meta",
                    [](Buffer &buffer, Buffer &other) {
                      auto other_ptr = std::shared_ptr<modelbox::Buffer>(
                          &other, [](void *data) {});
                      buffer.CopyMeta(other_ptr);
                    })
               .def("set",
                    [](Buffer &buffer, const std::string &key,
                       py::object &obj) { PythonBufferSet(buffer, key, obj); })
               .def("get", ModelboxPyApiSetUpBufferDefGet);

  ModelboxPyApiSetUpDataType(h);
}

void ModelboxPyApiSetUpBufferList(pybind11::module &m) {
  using namespace pybind11::literals;

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
      .def("push_back",
           [](BufferList &bl, Buffer &buffer) {
             auto new_buffer = std::make_shared<Buffer>(buffer);
             bl.PushBack(new_buffer);
           },
           py::keep_alive<1, 2>())
      .def("push_back",
           [](BufferList &bl, py::buffer b) {
             py::buffer_info info = b.request();
             std::vector<size_t> i_shape;
             for (auto &dim : info.shape) {
               i_shape.push_back(dim);
             }

             if (info.shape.size() == 0) {
               throw std::runtime_error("can not accpet empty numpy.");
             }

             size_t bytes = Volume(i_shape) * info.itemsize;
             auto buffer = std::make_shared<Buffer>(bl.GetDevice());
             buffer->BuildFromHost(info.ptr, bytes);
             buffer->Set("shape", i_shape);
             buffer->Set("type", TypeFromFormatStr(info.format));
             buffer->SetGetBufferType(modelbox::BufferEnumType::RAW);
             bl.PushBack(buffer);
           },
           py::keep_alive<1, 2>())
      .def("set",
           [](BufferList &bl, const std::string &key, py::object &obj) {
             for (auto &buffer : bl) {
               PythonBufferSet(*buffer, key, obj);
             }
           })
      .def("copy_meta", &modelbox::BufferList::CopyMeta)
      .def("__len__", [](const modelbox::BufferList &bl) { return bl.Size(); })
      .def("__iter__",
           [](const modelbox::BufferList &bl) {
             return py::make_iterator<
                 py::return_value_policy::reference_internal>(bl.begin(),
                                                              bl.end());
           },
           py::keep_alive<0, 1>())
      .def("__getitem__",
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
      .def("input", py::overload_cast<const std::string &>(
                        &modelbox::DataContext::Input, py::const_))
      .def("output", py::overload_cast<const std::string &>(
                         &modelbox::DataContext::Output))
      .def("external", &modelbox::DataContext::External)
      .def("event", &modelbox::DataContext::Event)
      .def("has_error", &modelbox::DataContext::HasError)
      .def("get_error", &modelbox::DataContext::GetError)
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
      .def("create_buffer",
           [](modelbox::FlowUnit &flow,
              py::buffer &b) -> std::shared_ptr<Buffer> {
             py::buffer_info info = b.request();
             std::vector<size_t> i_shape;
             for (auto &dim : info.shape) {
               i_shape.push_back(dim);
             }

             if (info.shape.size() == 0) {
               throw std::runtime_error("can not accpet empty numpy.");
             }

             size_t bytes = Volume(i_shape) * info.itemsize;
             auto buffer = std::make_shared<Buffer>(flow.GetBindDevice());
             if (bytes != 0) {
               buffer->BuildFromHost(info.ptr, bytes);
             }

             buffer->Set("shape", i_shape);
             buffer->Set("type", TypeFromFormatStr(info.format));
             buffer->SetGetBufferType(modelbox::BufferEnumType::RAW);
             return buffer;
           },
           py::keep_alive<0, 1>());
}

void ModelboxPyApiSetUpEngine(pybind11::module &m) {
  py::class_<modelbox::ModelBoxEngine,
             std::shared_ptr<modelbox::ModelBoxEngine>>(m, "ModelBoxEngine")
      .def(py::init<>())
      .def("init",
           [](ModelBoxEngine &env,
              std::shared_ptr<modelbox::Configuration> &config) {
             return env.Init(config);
           },
           py::call_guard<py::gil_scoped_release>())
      .def("init",
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
      .def("create_input",
           [](ModelBoxEngine &env, const std::set<std::string> &port_map) {
             py::gil_scoped_release release;
             return env.CreateInput(port_map);
           },
           py::keep_alive<0, 1>())
      .def("execute",
           [](ModelBoxEngine &env, const std::string &name,
              std::map<std::string, std::string> &config,
              std::map<std::string, std::shared_ptr<DataHandler>> &data) {
             py::gil_scoped_release release;
             return env.Execute(name, config, data);
           },
           py::keep_alive<0, 1>())
      .def("execute",
           [](ModelBoxEngine &env, const std::string &name,
              std::map<std::string, std::string> &config,
              std::shared_ptr<DataHandler> &data) {
             py::gil_scoped_release release;
             return env.Execute(name, config, data);
           },
           py::keep_alive<0, 1>());
}

void ModelboxPyApiSetUpFlowGraphDesc(pybind11::module &m) {
  py::class_<modelbox::FlowGraphDesc, std::shared_ptr<modelbox::FlowGraphDesc>>(
      m, "FlowGraphDesc")
      .def(py::init<>())
      .def("init",
           [](FlowGraphDesc &env,
              std::shared_ptr<modelbox::Configuration> &config) {
             return env.Init(config);
           },
           py::call_guard<py::gil_scoped_release>())
      .def("init",
           [](FlowGraphDesc &env,
              std::unordered_map<std::string, std::string> &config) {
             auto configuration = std::make_shared<modelbox::Configuration>();
             for (auto &iter : config) {
               configuration->SetProperty(iter.first, iter.second);
             }
             return env.Init(configuration);
           },
           py::call_guard<py::gil_scoped_release>())

      .def("bindinput", &modelbox::FlowGraphDesc::BindInput,
           py::call_guard<py::gil_scoped_release>())
      .def("bindoutput", &modelbox::FlowGraphDesc::BindOutput,
           py::call_guard<py::gil_scoped_release>())
      .def("addnode",
           [](FlowGraphDesc &env, const std::string &name,
              std::map<std::string, std::string> &config,
              std::map<std::string, std::shared_ptr<NodeDesc>> &data) {
             py::gil_scoped_release release;
             return env.AddNode(name, config, data);
           },
           py::keep_alive<0, 1>())
      .def("addnode",
           [](FlowGraphDesc &env, const std::string &name,
              std::map<std::string, std::string> &config,
              std::shared_ptr<NodeDesc> &data) {
             py::gil_scoped_release release;
             return env.AddNode(name, config, data);
           },
           py::keep_alive<0, 1>())
      .def("addnode",
           [](FlowGraphDesc &env,
              std::function<StatusCode(std::shared_ptr<DataContext>)> callback,
              std::vector<std::string> inputs, std::vector<std::string> outputs,
              std::shared_ptr<NodeDesc> datahandler) {
             py::gil_scoped_release release;
             return env.AddNode(callback, inputs, outputs, datahandler);
           },
           py::keep_alive<0, 1>())
      .def("addnode",
           [](FlowGraphDesc &env,
              std::function<StatusCode(std::shared_ptr<DataContext>)> callback,
              std::vector<std::string> inputs, std::vector<std::string> outputs,
              std::map<std::string, std::shared_ptr<NodeDesc>> &data) {
             py::gil_scoped_release release;

             return env.AddNode(callback, inputs, outputs, data);
           },
           py::keep_alive<0, 1>());
}

void ModelboxPyApiSetUpDataHandler(pybind11::module &m) {
  py::class_<modelbox::DataHandler, std::shared_ptr<modelbox::DataHandler>>(
      m, "DataHandler")
      .def(py::init<>())
      .def("close", &modelbox::DataHandler::Close,
           py::call_guard<py::gil_scoped_release>())
      .def("__iter__", [](DataHandler &data) -> DataHandler & { return data; },
           py::call_guard<py::gil_scoped_release>())
      .def("__next__",
           [](DataHandler &data) {
             py::gil_scoped_release release;
             auto buffer = data.GetData();
             if (buffer == nullptr) {
               throw pybind11::stop_iteration();
             }
             return buffer;
           },
           py::keep_alive<0, 1>())
      .def("__getitem__",
           [](DataHandler &data, const std::string &key) {
             auto sub_data = data.GetDataHandler(key);
             if (sub_data == nullptr) {
               throw pybind11::index_error();
             }
             return sub_data;
           },
           py::call_guard<py::gil_scoped_release>())
      .def("setmeta",
           py::overload_cast<const std::string &, const std::string &>(
               &modelbox::DataHandler::SetMeta),
           py::call_guard<py::gil_scoped_release>())
      .def("pushdata",
           py::overload_cast<std::shared_ptr<modelbox::Buffer> &,
                             const std::string>(
               &modelbox::DataHandler::PushData),
           py::call_guard<py::gil_scoped_release>())
      .def("pushdata",
           py::overload_cast<std::shared_ptr<DataHandler> &, const std::string>(
               &modelbox::DataHandler::PushData),
           py::call_guard<py::gil_scoped_release>())
      .def("get_datahandler", &modelbox::DataHandler::GetDataHandler,
           py::call_guard<py::gil_scoped_release>())
      .def("set_datahandler", &modelbox::DataHandler::SetDataHandler,
           py::call_guard<py::gil_scoped_release>());
}

void ModelboxPyApiSetUpNodeDesc(pybind11::module &m) {
  py::class_<modelbox::NodeDesc, std::shared_ptr<modelbox::NodeDesc>>(
      m, "NodeDesc")
      .def(py::init<>())
      .def("get_nodedesc", &modelbox::NodeDesc::GetNodeDesc,
           py::call_guard<py::gil_scoped_release>())
      .def("set_nodedesc", &modelbox::NodeDesc::SetNodeDesc,
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

}  // namespace modelbox