/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License") {
 * };
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

public class FlowUnitInput extends NativeObject {

    /**
     * constructor of flowunit input
     * @param name input name
     */
    public FlowUnitInput(String name) {
        setNativeHandle(FlowUnitInput_New(name));
    }

    /**
     * constructor of flowunit input
     * @param name input name
     * @param device_type device type
     */
    public FlowUnitInput(String name, String device_type) {
        setNativeHandle(FlowUnitInput_New(name, device_type));
    }

    /**
     * constructor of flowunit input
     * @param name input name
     * @param device_mem_flags device memory flags
     */
    public FlowUnitInput(String name, long device_mem_flags) {
        setNativeHandle(FlowUnitInput_New(name, device_mem_flags));
    }

    /**
     * constructor of flowunit input
     * @param name input name
     * @param device_type device type
     * @param device_mem_flags device memory flags
     */
    public FlowUnitInput(String name, String device_type, long device_mem_flags) {
        setNativeHandle(FlowUnitInput_New(name, device_type, device_mem_flags));
    }

    private native long FlowUnitInput_New(String name);

    private native long FlowUnitInput_New(String name, String device_type);

    private native long FlowUnitInput_New(String name, long device_mem_flags);

    private native long FlowUnitInput_New(String name, String device_type, long device_mem_flags);

}
