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

#ifndef MODELBOX_FLOW_UNIT_DATA_EXECUTOR_H_
#define MODELBOX_FLOW_UNIT_DATA_EXECUTOR_H_

#include <list>
#include <map>
#include <memory>
#include <utility>

#include "flowunit.h"
#include "modelbox/base/status.h"
#include "modelbox/buffer.h"
#include "modelbox/data_context.h"

namespace modelbox {

/**
 * @brief Bind flowunit with data context
 **/
class FlowUnitExecContext {
 public:
  FlowUnitExecContext(std::shared_ptr<FlowUnitDataContext> data_ctx);

  void SetFlowUnit(std::shared_ptr<FlowUnit> fu);

  const std::shared_ptr<FlowUnit> &GetFlowUnit();

  const std::shared_ptr<FlowUnitDataContext> &GetDataCtx();

 private:
  std::shared_ptr<FlowUnit> bind_fu_;
  std::shared_ptr<FlowUnitDataContext> data_ctx_;
};

/**
 * @brief data container for one data context
 * contains input, external and output
 **/
class FlowUnitExecData {
 public:
  enum DataType { IN_DATA, OUT_DATA };

  FlowUnitExecData(const std::shared_ptr<FlowUnit> &fu);

  virtual ~FlowUnitExecData();

  void ReserveCache(size_t buffer_count, DataType type = IN_DATA);

  void AppendToCache(const std::shared_ptr<FlowUnitExecData> &src,
                     size_t start_idx, size_t count, DataType type = IN_DATA);

  void FlushCache(DataType type = IN_DATA);

  void SetInData(const std::string &name,
                 const std::vector<std::shared_ptr<Buffer>> &buffer_list);

  std::shared_ptr<BufferListMap> GetInData();

  std::shared_ptr<BufferListMap> GetInDataForUser();

  std::shared_ptr<BufferList> GetInDataForUser(const std::string &name);

  std::shared_ptr<BufferListMap> GetOutData();

  std::shared_ptr<BufferList> GetOutData(const std::string &name);

  Status SetExternalData(
      const std::string &name,
      const std::vector<std::shared_ptr<Buffer>> &buffer_list);

  std::shared_ptr<BufferListMap> GetExternalData();

  std::shared_ptr<BufferListMap> GetExternalDataForUser();

  std::shared_ptr<BufferList> GetExternalDataForUser(const std::string &name);

  size_t GetInBufferNum();

  size_t GetExtBufferNum();

  size_t GetOutBufferNum(bool accumulate_all_port = false);

  void SetStatus(const Status &status);

  Status GetStatus() const;

  bool HasInData(const std::string &name) const;

  bool HasOutData(const std::string &name) const;

  bool HasExternData(const std::string &name) const;

  void SetupUserInput();

  Status SetupUserOutput(bool one_to_one, bool data_in_one_port);

  Status CheckStatus(bool one_to_one, bool data_in_one_port);

 private:
  void MakeCopyForUserOutput();

  void FillErrorOutput(size_t out_count, bool data_in_one_port);

  Status SaveProcessOneToOne(const std::shared_ptr<BufferListMap> &parent_data,
                             size_t data_count, bool data_in_one_port);

  Status SaveProcessNToM(const std::shared_ptr<BufferListMap> &parent_data);

  std::shared_ptr<FlowUnit> fu_;
  std::shared_ptr<BufferListMap> in_data_;
  std::shared_ptr<BufferListMap> in_data_for_user_;
  std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
      in_data_cache_;
  std::shared_ptr<BufferListMap> out_data_;
  std::unordered_map<std::string, std::vector<std::shared_ptr<Buffer>>>
      out_data_cache_;
  std::shared_ptr<BufferListMap> ext_data_;
  std::shared_ptr<BufferListMap> ext_data_for_user_;
  Status status_{STATUS_OK};
};

using FUExecContextList = std::list<std::shared_ptr<FlowUnitExecContext>>;
using BatchedFUExecDataCtx = std::vector<std::shared_ptr<ExecutorDataContext>>;
using BatchedFUExecDataCtxList = std::vector<BatchedFUExecDataCtx>;
using BatchedFUExecData = std::vector<std::shared_ptr<FlowUnitExecData>>;
using BatchedFUExecDataList = std::vector<BatchedFUExecData>;

/**
 * @brief This mapper contains all data for one flowunit
 * we have multi flowunit for one node, and each flowunit correspondes to
 * the node implementation in one device
 **/
class FlowUnitExecDataMapper {
 public:
  void AddExecCtx(const std::shared_ptr<FlowUnitExecContext> &exec_ctx);

  void LoadDataFromExecCtx();

  Status MapData(bool need_reshape, size_t batch_size, bool is_stream);

  Status MoveToTargetDevice(bool need_contiguous);

  void SetupUserInput();

  BatchedFUExecDataCtxList GetBatchedExecDataCtxList();

  Status CheckOutputDataNumber(bool data_in_one_port);

  Status CheckStatus(bool one_to_one, bool data_in_one_port);

  Status SetupUserOutput(bool one_to_one, bool data_in_one_port);

  Status SaveDataToExecCtx();

  void Clear();

 private:
  enum MapType { DIRECT_MAP, RESHAPE_NORMAL, RESHAPE_STREAM };

  bool NeedReshape(size_t batch_size);

  Status DirectMap();

  Status ReshapeNormal(size_t batch_size);

  void BuildMappedDataNormal(size_t batch_size);

  void FillMappedDataNormal(size_t batch_size);

  Status ReshapeStream(size_t batch_size);

  void BuildMappedDataStream();

  void FillMappedDataStream(size_t batch_size);

  Status MoveDataToTargetDevice(std::shared_ptr<BufferListMap> &data,
                                bool need_contiguous);

  Status WriteBackStream();

  Status WriteBackNormal();

  Status FillExecCtxOutput();

  Status CheckAllOutputNumEqual(const std::shared_ptr<FlowUnitExecData> &data,
                                bool data_in_one_port);

  Status CheckOutputNumEqualInput(const std::shared_ptr<FlowUnitExecData> &data,
                                  bool data_in_one_port);

  // data ctx list of same flowunit that come from node receive
  std::vector<std::shared_ptr<FlowUnitExecContext>> origin_exec_ctx_list_;
  // data from diff data ctx
  std::vector<std::shared_ptr<FlowUnitExecData>> origin_data_list_;
  std::vector<size_t> origin_shapes_;
  // after reshape and copy
  MapType map_type_;
  BatchedFUExecDataList mapped_data_list_;
  std::vector<std::vector<size_t>> mapped_shapes_;
  // data ctx list that fu process can see
  BatchedFUExecDataCtxList mapped_exec_data_ctx_list_;
};

using ExecViewVisitFunc =
    std::function<void(FlowUnit *flowunit, const BatchedFUExecDataCtxList
                                               &batched_exec_data_ctx_list)>;

/**
 * @brief contains multi flowunit data for one node
 * we use mapper to manage each flowunit data
 **/
class FlowUnitExecDataView {
 public:
  FlowUnitExecDataView(FUExecContextList exec_ctx_list);

  virtual ~FlowUnitExecDataView();

  Status LoadInputFromExecCtx(bool need_reshape, bool is_stream,
                              size_t batch_size, bool need_contiguous);

  const std::vector<FlowUnit *> &GetFlowUnits();

  const BatchedFUExecDataCtxList &GetFlowUnitProcessData(FlowUnit *flowunit);

  Status CheckOutputDataNumber(bool data_in_one_port);

  Status CheckStatus(bool one_to_one, bool data_in_one_port);

  Status SetupUserOutput(bool one_to_one, bool data_in_one_port);

  Status SaveOutputToExecCtx();

  void Clear();

 private:
  Status DevideExecCtxByFlowunit();

  FUExecContextList exec_ctx_list_;
  // data mapper for flowunits. each mapper contains all origin input data for
  // one flowunits
  std::unordered_map<FlowUnit *, std::shared_ptr<FlowUnitExecDataMapper>>
      mapper_of_flowunit_;
  // data wrapper for flowunits. each wrapper contains all process data for one
  // flowunit
  std::vector<FlowUnit *> flowunit_list_;
  std::unordered_map<FlowUnit *, BatchedFUExecDataCtxList> data_of_flowunit_;
  std::mutex data_of_flowunit_lock_;

  class LoadConfig {
   public:
    LoadConfig(bool need_reshape, bool is_stream, size_t batch_size,
               bool need_contiguous)
        : need_reshape_{need_reshape},
          is_stream_{is_stream},
          batch_size_{batch_size},
          need_contiguous_{need_contiguous} {}

    bool need_reshape_;
    bool is_stream_;
    size_t batch_size_;
    bool need_contiguous_;
  };

  Status DataLoadTask(
      const LoadConfig &cfg, FlowUnit *flowunit,
      const std::shared_ptr<FlowUnitExecDataMapper> &exec_data_mapper);

  Status PackLoadTasks(const LoadConfig &cfg,
                       std::vector<std::shared_ptr<Executor>> &executors,
                       std::vector<std::function<Status()>> &tasks);

  Status PackSaveTasks(std::vector<std::shared_ptr<Executor>> &executors,
                       std::vector<std::function<Status()>> &tasks);
};

class FlowUnitDataExecutor {
 public:
  FlowUnitDataExecutor(std::weak_ptr<Node> node_ref, size_t batch_size);

  virtual ~FlowUnitDataExecutor();

  virtual Status Process(const FUExecContextList &exec_ctx_list);

  Status DataCtxExecuteFunc(FlowUnit *flowunit,
                            const BatchedFUExecDataCtxList &process_data,
                            size_t data_ctx_idx);

  void SetNeedCheckOutput(bool need_check);

 private:
  Status LoadExecuteInput(const std::shared_ptr<Node> &node,
                          FlowUnitExecDataView &exec_view);

  Status Execute(FlowUnitExecDataView &exec_view);

  Status SaveExecuteOutput(const std::shared_ptr<Node> &node,
                           FlowUnitExecDataView &exec_view);

  std::weak_ptr<Node> node_ref_;
  size_t batch_size_;
  bool need_check_output_{false};
};

}  // namespace modelbox

#endif  // MODELBOX_FLOW_UNIT_DATA_EXECUTOR_H_