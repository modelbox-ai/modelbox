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


#include "modelbox/base/log.h"
#include "upsample-layer.h"
#include <typeinfo>

namespace nvinfer1
{
    __device__ int translate_idx(int index, int d1, int d2, int d3, int scale) {
        int x, y, z, w;
        w = index % d3;
        index = index/d3;
        z = index % d2;
        index = index/d2;
        y = index % d1;
        index = index/d1;
        x = index;
        w = w/scale;
        z = z/scale;
        d2 /= scale;
        d3 /= scale;
        return (((x*d1+y)*d2)+z)*d3+w;
    }
 
    //template <typename Dtype>
    __global__ void upscaleFp(const float *input, float *output,
                            int no_elements, int scale_factor, int d1, int d2, int d3) {
        int index = threadIdx.x + blockDim.x * blockIdx.x;
        if (index >= no_elements) return;
        int ipidx = translate_idx(index, d1, d2, d3, scale_factor);
        output[index]=input[ipidx];
    }
 
    __global__ void upscaleInt8(const uint8_t *input, uint8_t *output,
                              int no_elements, int scale_factor, int d1, int d2, int d3) {
        int index = threadIdx.x + blockDim.x * blockIdx.x;
        if (index >= no_elements) return;
        int ipidx = translate_idx(index, d1, d2, d3, scale_factor);
        output[index]=input[ipidx];
    }
 
 
    template <typename Dtype>
    void UpsampleLayerPlugin2::forwardGpu(const Dtype * input,Dtype * output,
                                         int N,int C,int H ,int W, cudaStream_t stream) {
        int numElem = N*C*H*W;
 
        if (typeid(Dtype) == typeid(float)) {
            upscaleFp<<<(numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream>>>((float *)input, (float *)output, numElem, mScale, C, H, W);
            return;
        } 
        
        if (typeid(Dtype) == typeid(uint8_t)) {
            upscaleInt8<<<(numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream>>>((uint8_t *)input, (uint8_t *)output, numElem, mScale, C, H, W);
            return;
        } 

        MBLOG_WARN << "upsample layer plugin forwardGpu only support float, int8";
    }
 
    int UpsampleLayerPlugin2::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        const int channel = mCHW.d[0];
        const int64_t in_height = mCHW.d[1];
        const int64_t in_width = mCHW.d[2];
        const int64_t out_height = mOutputHeight;
        const int64_t out_width = mOutputWidth;
        int totalElems = batchSize * in_height * in_width * channel;
 
        // Handle no-op resizes efficiently.
        if (out_height == in_height && out_width == in_width) {
            cudaMemcpyAsync(outputs[0], inputs[0], totalElems * type2size(mDataType), cudaMemcpyDeviceToDevice, stream);
            return 0;
        }
        cudaStreamSynchronize(stream);
 
        switch (mDataType)
        {
            case DataType::kFLOAT :
                forwardGpu<float>((const float *)inputs[0],(float *)outputs[0],batchSize,mCHW.d[0],mOutputHeight,mOutputWidth, stream);
                break;
            case DataType::kHALF:
                forwardGpu<__half>((const __half *)inputs[0],(__half *)outputs[0],batchSize,mCHW.d[0],mOutputHeight,mOutputWidth, stream);
                break;
            case DataType::kINT8:
                forwardGpu<u_int8_t>((const u_int8_t *)inputs[0],(u_int8_t *)outputs[0],batchSize,mCHW.d[0],mOutputHeight,mOutputWidth, stream);
                break;
            default:
                MBLOG_ERROR << "unsupport data type";
        }
 
        return 0;
    };
}