#ifndef INTERNAL_API_H
#define INTERNAL_API_H
#include "bigquant.h"
#include <stddef.h>
#include <stdint.h>

#ifdef WINDOWS
#define API_PREFIX __declspec(dllexport)
#else
#define API_PREFIX
#endif

extern "C" {

QuantizedConvOp *InternalQuantizedConvOpCreate();

void InternalQuantizedConvOpSetupConvParameter(QuantizedConvOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                               size_t group, size_t kernel_h, size_t kernel_w, size_t stride_h,
                                               size_t stride_w, size_t pad_h, size_t pad_w, size_t dialation_h,
                                               size_t dialation_w, size_t fusion_mask, CONV_ALGORITHM algo);

void InternalQuantizedConvOpInitWeight(QuantizedConvOp *p, float *weight);

void InternalQuantizedConvOpExecute(QuantizedConvOp *p, float *dst, float *data, float *bias, size_t batch_size,
                                    size_t channel_in, size_t height_in, size_t width_in);

void InternalQuantizedConvOpFree(QuantizedConvOp *p);

QuantizedFCOp *InternalQuantizedFCOpCreate();

void InternalQuantizedFCOpSetupFCParameter(QuantizedFCOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                           FC_ALGORITHM algo);

void InternalQuantizedFCOpInitWeight(QuantizedFCOp *p, float *weight);

void InternalQuantizedFCOpExecute(QuantizedFCOp *p, float *dst, float *data, float *bias, size_t batch_size,
                                  size_t channel_in);

void InternalQuantizedFCOpFree(QuantizedFCOp *p);

}
#endif
