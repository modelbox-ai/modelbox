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

#include "modelbox/output_buffer.h"

namespace modelbox {

OutputRings::OutputRings(OriginDataMap& output_map_buffer) {
  auto iter = output_map_buffer.begin();
  while (iter != output_map_buffer.end()) {
    auto key = iter->first;
    auto bufferlist = iter->second;
    std::vector<std::shared_ptr<IndexBuffer>> buffer_vector;

    std::shared_ptr<IndexBufferList> indexbuffer_list;

    // if the bufferlist is nullptr it's a condition
    if ((bufferlist->Size() == 1) && (bufferlist->At(0) == nullptr)) {
      indexbuffer_list = std::make_shared<IndexBufferList>(buffer_vector);
      output_map_.emplace(key, indexbuffer_list);
      iter++;
      continue;
    }

    for (uint32_t i = 0; i < bufferlist->Size(); i++) {
      auto buffer = std::make_shared<IndexBuffer>(bufferlist->At(i));
      buffer_vector.push_back(buffer);
    }
    indexbuffer_list = std::make_shared<IndexBufferList>(buffer_vector);
    buffer_matrix_.push_back(indexbuffer_list);
    output_map_.emplace(key, indexbuffer_list);
    iter++;
  }
}

OutputRings::~OutputRings() {}

Status OutputRings::IsValid() {
  // check the output is valid
  if ((buffer_matrix_.size() != output_map_.size()) &&
      (buffer_matrix_.size() != 1)) {
    return {STATUS_INVALID, "The buffer matrix size is not correct"};
  }

  uint32_t default_vector_size = 0;
  uint32_t zero_count = 0;

  for (uint32_t i = 0; i < buffer_matrix_.size(); i++) {
    auto vector_size = buffer_matrix_[i]->GetBufferNum();

    if (vector_size != 0) {
      if (default_vector_size == 0) {
        default_vector_size = vector_size;
      }
    } else {
      zero_count++;
    }

    if ((default_vector_size != vector_size) && (vector_size != 0)) {
      return {STATUS_INVALID, "The buffer matrix size is not equal"};
    }
  }

  if ((zero_count != 0) && (zero_count != buffer_matrix_.size() - 1) &&
      (zero_count != buffer_matrix_.size())) {
    return {STATUS_INVALID, "The buffer matrix zero count is not correct"};
  }
  return STATUS_SUCCESS;
}

std::shared_ptr<IndexBufferList> OutputRings::GetOneBufferList() {
  return buffer_matrix_[0];
}

Status OutputRings::AppendOutputMap(OutputIndexBuffer* output_buffer) {
  if (output_buffer == nullptr) {
    return {STATUS_INVALID, "the output map is nullptr"};
  }
  auto iter = output_buffer->begin();
  while (iter != output_buffer->end()) {
    auto key = iter->first;
    if (output_map_.find(key) == output_map_.end()) {
      output_buffer->clear();
      return {STATUS_INVALID, "key not find in the map: " + key};
    }
    auto indexbuffer_list = output_map_.find(key)->second;
    for (uint32_t i = 0; i < indexbuffer_list->GetBufferNum(); i++) {
      auto index_buffer = indexbuffer_list->GetBuffer(i);

      if (index_buffer->GetBufferPtr() != nullptr ||
          index_buffer->IsPlaceholder()) {
        output_buffer->find(key)->second.push_back(index_buffer);
      }
    }
    iter++;
  }
  return STATUS_SUCCESS;
}

std::shared_ptr<IndexBufferList> OutputRings::GetBufferList(
    const std::string& key) {
  if (output_map_.find(key) == output_map_.end()) {
    return nullptr;
  }
  return output_map_[key];
}

Status OutputRings::BroadcastMetaToAll() {
  if (buffer_matrix_.size() == 0) {
    return STATUS_INVALID;
  }

  auto ring_buf_list = buffer_matrix_[0];
  if (ring_buf_list->GetBufferNum() == 0) {
    return STATUS_INVALID;
  }

  for (uint32_t j = 0; j < ring_buf_list->GetBufferNum(); j++) {
    for (uint32_t i = 1; i < buffer_matrix_.size(); i++) {
      auto origin_buffer = ring_buf_list->GetBuffer(j);
      auto now_buffer = buffer_matrix_[i]->GetBuffer(j);
      if (!origin_buffer->CopyMetaTo(now_buffer)) {
        return {STATUS_INVALID, "copy meta to other buffer failed"};
      }
    }
  }
  return STATUS_SUCCESS;
}

std::shared_ptr<FlowUnitError> OutputRings::GetError() {
  for (uint32_t i = 0; i < buffer_matrix_.size(); i++) {
    auto err_index = buffer_matrix_[i]->GetDataErrorIndex();
    auto buffer_num = buffer_matrix_[i]->GetBufferNum();
    if (buffer_num == 0) {
      return nullptr;
    }

    if (buffer_matrix_[i]->GetBuffer(0)->GetSameLevelGroup() == nullptr) {
      return nullptr;
    }

    auto stream_bg =
        buffer_matrix_[i]->GetBuffer(0)->GetSameLevelGroup()->GetGroup();
    if (stream_bg->GetDataError(err_index) != nullptr) {
      return stream_bg->GetDataError(err_index);
    }
  }

  return nullptr;
}

void OutputRings::BackfillOneOutput(std::set<uint32_t> id_set) {
  if (id_set.empty()) {
    return;
  }
  int i = 0;
  for (auto& index_buffer_list : buffer_matrix_) {
    if (i == 0) {
      index_buffer_list->BackfillBuffer(id_set);
    } else {
      index_buffer_list->BackfillBuffer(id_set, false);
    }
    i++;
  }
}

void OutputRings::BackfillOutput(std::set<uint32_t> id_set) {
  if (id_set.empty()) {
    return;
  }

  for (auto& index_buffer_list : buffer_matrix_) {
    index_buffer_list->BackfillBuffer(id_set);
  }
}

void OutputRings::Clear() {
  buffer_matrix_.clear();
  output_map_.clear();
}

bool OutputRings::IsEmpty() {
  if ((buffer_matrix_.size() == 0) && (output_map_.size() == 0)) {
    return true;
  }
  return false;
}

void OutputRings::SetAllPortError(const std::shared_ptr<FlowUnitError> error) {
  for (uint32_t i = 0; i < buffer_matrix_.size(); i++) {
    if (buffer_matrix_[i]->GetBufferNum() == 0) {
      break;
    }
    auto err_index = buffer_matrix_[i]->GetDataErrorIndex();
    auto stream_bg =
        buffer_matrix_[i]->GetBuffer(0)->GetSameLevelGroup()->GetGroup();
    stream_bg->SetDataError(err_index, error);
  }
}

}  // namespace modelbox
