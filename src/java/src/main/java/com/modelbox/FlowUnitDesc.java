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

/**
 * Flowunit description
 */
public class FlowUnitDesc extends NativeObject {
    public enum FlowOutputType {
        ORIGIN, EXPAND, COLLAPSE,
    }

    public enum FlowType {
        STREAM, NORMAL,
    }

    public enum ConditionType {
        NONE, IF_ELSE,
    }

    public enum LoopType {
        NOT_LOOP, LOOP,
    }

    public FlowUnitDesc() {
        setNativeHandle(FlowUnitDescNew());
    }

    /**
     * Get flowunit name.
     * @return String
     */
    public String GetFlowUnitName() {
        return FlowUnitDescGetFlowUnitName();
    }

    /**
     * Get flowunit type.
     * @return flowunit type
     */
    public String GetFlowUnitType() {
        return FlowUnitDescGetFlowUnitType();
    }

    /**
     * Get flowunit alias name.
     * @return flowunit alias name
     */
    public String GetFlowUnitAliasName() {
        return FlowUnitDescGetFlowUnitAliasName();
    }

    /**
     * Get flowunit argument.
     * @return flowunit argument
     */
    public String GetFlowUnitArgument() {
        return FlowUnitDescGetFlowUnitArgument();
    }

    /**
     * Set flowunit output name.
     * @param flowunit output name
     */
    public void SetFlowUnitName(String flowunit_name) {
        FlowUnitDescSetFlowUnitName(flowunit_name);
    }

    /**
     * Set flowunit type.
     * @param flowunit type
     */
    public void SetFlowUnitType(String flowunit_type) {
        FlowUnitDescSetFlowUnitType(flowunit_type);
    }

    /**
     * Add input to flowunit
     * @param flowunit_input input
     */
    public void AddFlowUnitInput(FlowUnitInput flowunit_input) throws ModelBoxException {
        FlowUnitDescAddFlowUnitInput(flowunit_input);
    }

    /**
     * Add output to flowunit
     * @param flowunit_output output
     */
    public void AddFlowUnitOutput(FlowUnitOutput flowunit_output) throws ModelBoxException {
        FlowUnitDescAddFlowUnitOutput(flowunit_output);
    }

    /**
     * Set flowunit condition type
     * @param condition_type condition type
     */
    public void SetConditionType(ConditionType condition_type) {
        FlowUnitDescSetConditionType(condition_type.ordinal());
    }

    /**
     * Set flowunit loop type
     * @param loop_type loop type
     */
    public void SetLoopType(LoopType loop_type) {
        FlowUnitDescSetLoopType(loop_type.ordinal());
    }

    /**
     * Set flowunit output type
     * @param output_type output type
     */
    public void SetOutputType(FlowOutputType output_type) {
        FlowUnitDescSetOutputType(output_type.ordinal());
    }

    /**
     * Set flowunit type
     * @param flow_type flow type
     */
    public void SetFlowType(FlowType flow_type) {
        FlowUnitDescSetFlowType(flow_type.ordinal());
    }

    /**
     * Set flowunit same count
     * @param is_stream_same_count is same count
     */
    public void SetStreamSameCount(boolean is_stream_same_count) {
        FlowUnitDescSetStreamSameCount(is_stream_same_count);
    }

    /**
     * Set flowunit input contiguous
     * @param is_input_contiguous flowunit input contiguous
     */
    public void SetInputContiguous(boolean is_input_contiguous) {
        FlowUnitDescSetInputContiguous(is_input_contiguous);
    }

    /**
     * Set flowunit is source nice
     * @param is_resource_nice flowunit is source nice
     */
    public void SetResourceNice(boolean is_resource_nice) {
        FlowUnitDescSetResourceNice(is_resource_nice);
    }

    /**
     * Set flowunit is collapse
     * @param is_collapse_all flowunit is collapse
     */
    public void SetCollapseAll(boolean is_collapse_all) {
        FlowUnitDescSetCollapseAll(is_collapse_all);
    }

    /**
     * Set flowunit is visible exception
     * @param is_exception_visible flowunit is visible exception
     */
    public void SetExceptionVisible(boolean is_exception_visible) {
        FlowUnitDescSetExceptionVisible(is_exception_visible);
    }

    /**
     * Set flowunit description info
     * @param description flowunit description info
     */
    public void SetDescription(String description) {
        FlowUnitDescSetDescription(description);
    }

    /**
     * Set flowunit max batch size
     * @param max_batch_size flowunit max batch size
     */
    public void SetMaxBatchSize(long max_batch_size) {
        FlowUnitDescSetMaxBatchSize(max_batch_size);
    }

    /**
     * Set flowunit default batch size
     * @param default_batch_size flowunit max batch size
     */
    public void SetDefaultBatchSize(long default_batch_size) {
        FlowUnitDescSetDefaultBatchSize(default_batch_size);
    }

    private native long FlowUnitDescNew();

    private native String FlowUnitDescGetFlowUnitName();

    private native String FlowUnitDescGetFlowUnitType();

    private native String FlowUnitDescGetFlowUnitAliasName();

    private native String FlowUnitDescGetFlowUnitArgument();

    private native void FlowUnitDescSetFlowUnitName(String flowunit_name);

    private native void FlowUnitDescSetFlowUnitType(String flowunit_type);

    private native void FlowUnitDescAddFlowUnitInput(FlowUnitInput flowunit_input);

    private native void FlowUnitDescAddFlowUnitOutput(FlowUnitOutput flowunit_output);

    private native void FlowUnitDescSetConditionType(long condition_type);

    private native void FlowUnitDescSetLoopType(long loop_type);

    private native void FlowUnitDescSetOutputType(long output_type);

    private native void FlowUnitDescSetFlowType(long flow_type);

    private native void FlowUnitDescSetStreamSameCount(boolean is_stream_same_count);

    private native void FlowUnitDescSetInputContiguous(boolean is_input_contiguous);

    private native void FlowUnitDescSetResourceNice(boolean is_resource_nice);

    private native void FlowUnitDescSetCollapseAll(boolean is_collapse_all);

    private native void FlowUnitDescSetExceptionVisible(boolean is_exception_visible);

    private native void FlowUnitDescSetDescription(String description);

    private native void FlowUnitDescSetMaxBatchSize(long max_batch_size);

    private native void FlowUnitDescSetDefaultBatchSize(long default_batch_size);
}
