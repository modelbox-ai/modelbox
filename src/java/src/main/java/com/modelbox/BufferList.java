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

package com.modelbox;


import java.nio.ByteBuffer;
import java.util.Iterator;

/**
 * Modelbox Bufferlist
 */

public class BufferList extends NativeObject implements Iterable<Buffer> {

  class BufferListIterator implements Iterator<Buffer> {

    @Override
    public boolean hasNext() {
      if (index < bufferList.size()) {
        return true;
      }
      return false;
    }

    @Override
    public Buffer next() {
      Buffer buff = bufferList.at(index);
      index++;
      return buff;
    }

    protected void SetBufferList(BufferList list) {
      this.bufferList = list;
    }

    private BufferList bufferList;
    int index = 0;
  }

  public Iterator<Buffer> iterator() {
    BufferListIterator itr = new BufferListIterator();
    itr.SetBufferList(this);
    return itr;
  }

  /**
   * modelbox buffer list.
   * Create Bufferlist from external data map
   */
  private BufferList() {

  }

  /**
   * Builder buffer, create memory
   * @param sizeList buffer size list
   */
  public void build(int[] sizeList) {
    BufferListBuild(sizeList);
  }

  /**
   * Get buffer at index
   * @param index position of buffer
   * @return buffer
   */
  public Buffer at(long index) {
    return BufferListAt(index);
  }

  /**
   * Get number of buffer in bufferlist
   * @return number of buffer
   */
  public long size() {
    return BufferListSize();
  }

  /**
   * Push new buffer to buffer list
   * @param buffer pointer to buffer
   */
  public void pushBack(Buffer buffer) {
    BufferListPushBack(buffer);
  }

  /**
   * Push new data to buffer list
   * @param data pointer to data
   */
  public void pushBack(byte[] data) {
    BufferListPushBack(data);
  }

  /**
   * Assign buffer list
   * @param buffers buffer list to assign
   */
  public void assign(Buffer[] buffers) {
    BufferListAssign(buffers);
  }

  /**
   * Get buffer data pointer from begining
   * @return buffer data pointer from begining
   */
  public byte[] getData() {
    return BufferListGetData();
  }

  /**
   * Get buffer data pointer from begining
   * <p>WARNING: Do not hold ByteBuffer beyond the scope, otherwise it will cause dangling pointers and trigger coredump
   * @return buffer data pointer from begining
   */
  public ByteBuffer getDirectData() {
    return BufferListGetDirectData();
  }

  /**
   * Get buffer data pointer from begining
   * WARNING: Do not hold ByteBuffer beyond the scope, otherwise it will cause dangling pointers and trigger coredump
   * @param index position of buffer
   * @return buffer data pointer from begining
   */
  public ByteBuffer getDirectData(int index) {
    return BufferListGetDirectData(index);
  }

  /**
   * Get device of buffer list
   * @return pointer to device
   */
  public Device getDevice() {
    return BufferListGetDevice();
  }

  /**
   * Reset buffer list
   * @return reset result
   */
  public void reset() {
    BufferListReset();
  }

  private native void BufferListBuild(int[] sizeList);

  private native Buffer BufferListAt(long index);

  private native long BufferListSize();

  private native void BufferListPushBack(Buffer buffer);

  private native void BufferListPushBack(byte[] data);

  private native void BufferListAssign(Buffer[] buffers);

  private native byte[] BufferListGetData();

  private native ByteBuffer BufferListGetDirectData();

  private native ByteBuffer BufferListGetDirectData(int index);

  private native Device BufferListGetDevice();

  private native void BufferListReset();
}
