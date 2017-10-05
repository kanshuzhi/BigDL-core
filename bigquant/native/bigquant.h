#ifndef BIGQUANT_H
#define BIGQUANT_H
#include <stddef.h>
#include <stdint.h>

typedef enum LAYOUT { NCHW = 0, NHWC = 1 } LAYOUT;
typedef enum CONV_ALGORITHM { AUTO_SELECT_CONV = 0, SHUFFLE_CONV = 1 } CONV_ALGORITHM;
typedef enum FC_ALGORITHM { AUTO_SELECT_FC = 0, SHUFFLE_FC = 1 } FC_ALGORITHM;

struct QuantizedConvOp;
typedef struct QuantizedConvOp QuantizedConvOp;

struct QuantizedFCOp;
typedef struct QuantizedFCOp QuantizedFCOp;

#ifdef WINDOWS
#define API_PREFIX __declspec(dllexport)
#else
#define API_PREFIX
#endif

#ifdef __cplusplus
extern "C" {
#endif

API_PREFIX int ManualRuntimeLoadLib(char *path);

API_PREFIX QuantizedConvOp *QuantizedConvOpCreate();

API_PREFIX void QuantizedConvOpSetupConvParameter(QuantizedConvOp *p, LAYOUT layout, size_t channel_out,
                                                  size_t channel_in, size_t group, size_t kernel_h, size_t kernel_w,
                                                  size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                                  size_t dialation_h, size_t dialation_w, size_t fusion_mask,
                                                  CONV_ALGORITHM algo);

API_PREFIX void QuantizedConvOpInitWeight(QuantizedConvOp *p, float *weight);

API_PREFIX void QuantizedConvOpExecute(QuantizedConvOp *p, float *dst, float *data, float *bias, size_t batch_size,
                                       size_t channel_in, size_t height_in, size_t width_in);

API_PREFIX void QuantizedConvOpFree(QuantizedConvOp *p);

API_PREFIX QuantizedFCOp *QuantizedFCOpCreate();

API_PREFIX void QuantizedFCOpSetupFCParameter(QuantizedFCOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                              FC_ALGORITHM algo);

API_PREFIX void QuantizedFCOpInitWeight(QuantizedFCOp *p, float *weight);

API_PREFIX void QuantizedFCOpExecute(QuantizedFCOp *p, float *dst, float *data, float *bias, size_t batch_size,
                                     size_t channel_in);

API_PREFIX void QuantizedFCOpFree(QuantizedFCOp *p);

#ifdef __cplusplus
}
#endif

#endif
