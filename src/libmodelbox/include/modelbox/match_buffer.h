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


#ifndef MODELBOX_MATCH_BUFFER_H_
#define MODELBOX_MATCH_BUFFER_H_
#include "modelbox/index_buffer.h"
namespace modelbox {

class MatchBuffer {
 public:
  /**
   * @brief Construct a new MatchBuffer object
   *
   * @param match_num the input num
   */
  MatchBuffer(uint32_t match_num);

  /**
   * @brief destructor
   */
  virtual ~MatchBuffer();

  /**
   * @brief Is the MatchBuffer contain all the buffer it need
   *
   * @return true
   * @return false
   */
  bool IsMatch();

  /**
   * @brief Get the MatchBuffer order in its parent group
   *
   * @return uint32_t const
   */
  uint32_t const GetOrder();

  /**
   * @brief Get the MatchBuffer parent group sum
   *
   * @param sum the group sum
   * @return Status {status} if success return STATUS_SUCCESS if There is none
   * return STATUS_NOTFOUND
   */
  Status GetGroupSum(uint32_t* sum);

  /**
   * @brief Set the Buffer object
   *
   * @param key {string} the buffer from which port
   * @param buffer {buffer} the actual buffer
   * @return true success to insert the buffer
   * @return false the key already exist or the buffer is not in the same group
   */
  bool SetBuffer(std::string key, std::shared_ptr<IndexBuffer> buffer);

  /**
   * @brief Get the Buffer object
   *
   * @param key {string} the buffer from which port
   * @return std::shared_ptr<IndexBuffer>  return nullptr if key is not exist
   */
  std::shared_ptr<IndexBuffer> GetBuffer(std::string key);

  std::shared_ptr<BufferGroup> GetBufferGroup();

 private:
  // record the MatchBuffer's group
  std::shared_ptr<BufferGroup> group_ptr_;
  uint32_t match_num_;
  std::map<std::string, std::shared_ptr<IndexBuffer>> match_buffer_;
};

}  // namespace modelbox
#endif