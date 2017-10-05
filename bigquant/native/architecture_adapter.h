#ifndef ARCHITECTURE_ADAPTER_H
#define ARCHITECTURE_ADAPTER_H

#include "base.h"
#include "common.h"
#include "ops/ops.h"
#include <mutex>

struct BaseArchitectureAdapter {
 public:
  typedef void (*shuffle_gemm_function)(int8_t *, uint8_t *, float *, size_t, size_t, size_t, float *, float *,
                     float *, float *, float *, size_t, size_t,
                     size_t, size_t, size_t, size_t,
                     float, size_t, size_t);

  typedef void (*stream_quantize_function) (uint8_t *, float *, const float, const float,
                                  const size_t, const size_t, const size_t);

  typedef void (*shuffle_im2col_function) (float *, size_t, size_t, size_t,
                                     size_t, size_t, size_t, size_t, size_t,
                                     size_t, size_t, size_t, size_t,
                                     size_t, uint8_t *[], float *[], float *[], float *[],
                                     float *, float, bool);

  static BaseArchitectureAdapter& GetInstance() {
    std::lock_guard<std::mutex> scope_lock(lock);
    if (p == NULL) {
      p = InitInstance();
    }
    return *p;
  }

  virtual shuffle_gemm_function GetShuffleGemmFunction(bool has_spatial_dim, size_t batch, LAYOUT layout) = 0;

  virtual stream_quantize_function GetStreamQuantizeFunction() = 0;

  virtual shuffle_im2col_function GetShuffleIm2colFunction(LAYOUT layout) = 0;

  BaseArchitectureAdapter(const BaseArchitectureAdapter&) = delete;

  BaseArchitectureAdapter& operator=(const BaseArchitectureAdapter&) = delete;

  static void Free() {
    delete p;
    p = NULL;
  }

 protected:
  BaseArchitectureAdapter() {

  }

  virtual ~BaseArchitectureAdapter() {

  }

  static BaseArchitectureAdapter* InitInstance() {
    return NULL;
  }

  static BaseArchitectureAdapter *p;
  static std::mutex lock;
};

BaseArchitectureAdapter *BaseArchitectureAdapter::p = NULL;
std::mutex BaseArchitectureAdapter::lock;


#if defined(AVX512)
struct AVX512ArchitectureAdapter : BaseArchitectureAdapter {

 public:

  static BaseArchitectureAdapter& GetInstance() {
    std::lock_guard<std::mutex> scope_lock(lock);
    if (p == NULL) {
      p = InitInstance();
    }
    return *p;
  }

  shuffle_gemm_function GetShuffleGemmFunction(bool has_spatial_dim, size_t batch_size, LAYOUT gemm_layout) {
    if (has_spatial_dim || batch_size > 4) {
      if (gemm_layout == NCHW) {
        return shuffle::ConvShuffleGEMM<8, 8, 8, NCHW>;
      } else {
        return shuffle::ConvShuffleGEMM<8, 8, 8, NHWC>;
      }
    } else {
      if (gemm_layout == NCHW) {
        return shuffle::ConvShuffleGEMM<8, 8, 8, NCHW>;
      } else {
        return shuffle::ConvShuffleGEMM<8, 8, 8, NHWC>;
      }
    }
  }

  stream_quantize_function GetStreamQuantizeFunction() {
    return AVX512Kernel8StreamQuantize;
  }

  shuffle_im2col_function GetShuffleIm2colFunction(LAYOUT layout) {
    if (layout == NCHW) {
      return shuffle::PadQuantizeShuffleIm2colWrapper<NCHW, 8, 8, AVX512Kernel8StreamQuantize>;
    } else {
      return shuffle::PadQuantizeShuffleIm2colWrapper<NHWC, 8, 8, AVX512Kernel8StreamQuantize>;
    }
  }

 protected:

  static BaseArchitectureAdapter* InitInstance() {
    return new AVX512ArchitectureAdapter;
  }

};

#elif defined(__AVX2__)

struct AVX2ArchitectureAdapter : BaseArchitectureAdapter {

 public:

  static BaseArchitectureAdapter& GetInstance() {
    std::lock_guard<std::mutex> scope_lock(lock);
    if (p == NULL) {
      p = InitInstance();
    }
    return *p;
  }

  virtual shuffle_gemm_function GetShuffleGemmFunction(bool has_spatial_dim, size_t batch_size, LAYOUT gemm_layout) {
    if (has_spatial_dim || batch_size > 4) {
      if (gemm_layout == NCHW) {
        return shuffle::ConvShuffleGEMM<4, 8, 8, NCHW>;
      } else {
        return shuffle::ConvShuffleGEMM<4, 8, 8, NHWC>;
      }
    } else {
      if (gemm_layout == NCHW) {
        return shuffle::ConvShuffleGEMM<4, 8, 8, NCHW>;
      } else {
        return shuffle::ConvShuffleGEMM<4, 8, 8, NHWC>;
      }
    }
  }

  stream_quantize_function GetStreamQuantizeFunction() {
    return AVX2Kernel8StreamQuantize;
  }

  shuffle_im2col_function GetShuffleIm2colFunction(LAYOUT layout) {
    if (layout == NCHW) {
      return shuffle::PadQuantizeShuffleIm2colWrapper<NCHW, 8, 8, AVX2Kernel8StreamQuantize>;
    } else {
      return shuffle::PadQuantizeShuffleIm2colWrapper<NHWC, 8, 8, AVX2Kernel8StreamQuantize>;
    }
  }

 protected:

  static BaseArchitectureAdapter* InitInstance() {
    return new AVX2ArchitectureAdapter;
  }

};

#else

struct SSE42ArchitectureAdapter : BaseArchitectureAdapter {

 public:

  static BaseArchitectureAdapter& GetInstance() {
    std::lock_guard<std::mutex> scope_lock(lock);
    if (p == NULL) {
      p = InitInstance();
    }
    return *p;
  }

  virtual shuffle_gemm_function GetShuffleGemmFunction(bool has_spatial_dim, size_t batch_size, LAYOUT gemm_layout) {
    if (has_spatial_dim || batch_size > 4) {
      if (gemm_layout == NCHW) {
        return shuffle::ConvShuffleGEMM<2, 2, 16, NCHW>;
      } else {
        return shuffle::ConvShuffleGEMM<2, 2, 16, NHWC>;
      }
    } else {
      if (gemm_layout == NCHW) {
        return shuffle::ConvShuffleGEMM<2, 2, 16, NCHW>;
      } else {
        return shuffle::ConvShuffleGEMM<2, 2, 16, NHWC>;
      }
    }
  }

  stream_quantize_function GetStreamQuantizeFunction() {
    return SSE42Kernel16StreamQuantize;
  }

  shuffle_im2col_function GetShuffleIm2colFunction(LAYOUT layout) {
    if (layout == NCHW) {
      return shuffle::PadQuantizeShuffleIm2colWrapper<NCHW, 2, 16, SSE42Kernel16StreamQuantize>;
    } else {
      return shuffle::PadQuantizeShuffleIm2colWrapper<NHWC, 2, 16, SSE42Kernel16StreamQuantize>;
    }
  }

 protected:

  static BaseArchitectureAdapter* InitInstance() {
    return new SSE42ArchitectureAdapter;
  }

};
#endif

BaseArchitectureAdapter& MakeArchitectuerAdapter() {
#if defined(AVX512)
  return AVX512ArchitectureAdapter::GetInstance();
#elif defined(__AVX2__)
  return AVX2ArchitectureAdapter::GetInstance();
#else
  return SSE42ArchitectureAdapter::GetInstance();
#endif
}

#endif
