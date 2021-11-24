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


#ifndef MODELBOX_LINK_MAP_BUFFER_H_
#define MODELBOX_LINK_MAP_BUFFER_H_

#include <unordered_map>

#include "modelbox/index_buffer.h"

namespace modelbox {
using OutputIndexBuffer =
    std::unordered_map<std::string, std::vector<std::shared_ptr<IndexBuffer>>>;

using OriginDataMap =
    std::unordered_map<std::string, std::shared_ptr<BufferList>>;

// the ring buffer lists we can get the content from map or the list
class OutputRings {
 public:
  /**
   * @brief Construct a new Output Rings object
   * @param output_map_buffer
   */
  OutputRings(OriginDataMap& output_map_buffer);

  /**
   * @brief Destruct of OutputRings
   */
  virtual ~OutputRings();

  /**
   * @brief Init the OutputRings
   *
   * @return Status {status} return STATUS_SUCCESS if success to init the
   * OutputRings
   */
  Status IsValid();

  /**
   * @brief Get the First Buffer List object
   *
   * @return std::shared_ptr<RingBufferList>  if the OutputRings init failed
   * return nullptr
   */
  std::shared_ptr<IndexBufferList> GetOneBufferList();

  /**
   * @brief After Init Sucess we can broadcast the meta to all the data
   *
   * @return Status {status} return STATUS_SUCCESS if success to
   * BroadcastMetaToRing
   */
  Status BroadcastMetaToAll();

  /**
   * @brief Append the current OutputRings to the targe OutputIndexBuffer
   *
   * @param map targe OutputIndexBuffer
   * @return Status {status} return STATUS_SUCCESS if success to AppendOutputMap
   */
  Status AppendOutputMap(OutputIndexBuffer* map);

  void BackfillOutput(std::set<uint32_t> id_set);

  void BackfillOneOutput(std::set<uint32_t> id_set);

  std::shared_ptr<IndexBufferList> GetBufferList(const std::string& key);

  std::shared_ptr<FlowUnitError> GetError();

  void SetAllPortError(const std::shared_ptr<FlowUnitError> error);

  void Clear();

  bool IsEmpty();

 private:
  std::vector<std::shared_ptr<IndexBufferList>> buffer_matrix_;
  std::unordered_map<std::string, std::shared_ptr<IndexBufferList>> output_map_;
  std::shared_ptr<FlowUnitError> error_;
};

}  // namespace modelbox

#endif