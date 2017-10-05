#include "internal_api.h"
#include "base.h"
#include "common.h"
#include "alloc.h"
#include "model.h"
#include "ops/ops.h"
#include "nn/convolution_op.h"
#include "nn/fc_op.h"

// The following is Descriptor based APU
QuantizedConvOp *InternalQuantizedConvOpCreate() {
  ConvOp *p = new ConvOp();
  return reinterpret_cast<QuantizedConvOp *>(p);
}

void InternalQuantizedConvOpSetupConvParameter(QuantizedConvOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                               size_t group, size_t kernel_h, size_t kernel_w, size_t stride_h,
                                               size_t stride_w, size_t pad_h, size_t pad_w, size_t dialation_h,
                                               size_t dialation_w, size_t fusion_mask, CONV_ALGORITHM algo) {
  reinterpret_cast<ConvOp *>(p)->SetupConvolutionParameter(layout, channel_out, channel_in, group, kernel_h, kernel_w,
                                                           stride_h, stride_w, pad_h, pad_w, dialation_h, dialation_w,
                                                           fusion_mask, algo);
}

void InternalQuantizedConvOpInitWeight(QuantizedConvOp *p, float *weight) {
  reinterpret_cast<ConvOp *>(p)->InitWeight(weight);
}

void InternalQuantizedConvOpExecute(QuantizedConvOp *p, float *dst, float *data, float *bias, size_t batch_size,
                                    size_t channel_in, size_t height_in, size_t width_in) {
  reinterpret_cast<ConvOp *>(p)->Execute(dst, data, bias, batch_size, channel_in, height_in, width_in);
}
void InternalQuantizedConvOpFree(QuantizedConvOp *p) {
  delete reinterpret_cast<ConvOp *>(p);
}

QuantizedFCOp *InternalQuantizedFCOpCreate() {
  FCOp *p = new FCOp();
  return reinterpret_cast<QuantizedFCOp *>(p);
}

void InternalQuantizedFCOpSetupFCParameter(QuantizedFCOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                           FC_ALGORITHM algo) {
  reinterpret_cast<FCOp *>(p)->SetupFCKernelParameter(layout, channel_out, channel_in, algo);
}

void InternalQuantizedFCOpInitWeight(QuantizedFCOp *p, float *weight) {
  reinterpret_cast<FCOp *>(p)->InitWeight(weight);
}

void InternalQuantizedFCOpExecute(QuantizedFCOp *p, float *dst, float *data, float *bias, size_t batch_size,
                                  size_t channel_in) {
  reinterpret_cast<FCOp *>(p)->Execute(dst, data, bias, batch_size, channel_in);
}

void InternalQuantizedFCOpFree(QuantizedFCOp *p) {
  delete reinterpret_cast<FCOp *>(p);
}

