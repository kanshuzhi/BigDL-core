#if defined(WINDOWS)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include "bigquant.h"
#include "base.h"
#include "common.h"

#if defined(WINDOWS)
HINSTANCE handler = NULL;
#else
void *handler = NULL;
#endif

QuantizedConvOp *(*QuantizedConvOpCreateRT)();

void (*QuantizedConvOpSetupConvParameterRT)(QuantizedConvOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                            size_t group, size_t kernel_h, size_t kernel_w, size_t stride_h,
                                            size_t stride_w, size_t pad_h, size_t pad_w, size_t dialation_h,
                                            size_t dialation_w, size_t fusion_mask, CONV_ALGORITHM algo);

void (*QuantizedConvOpInitWeightRT)(QuantizedConvOp *p, float *weight);

void (*QuantizedConvOpExecuteRT)(QuantizedConvOp *p, float *dst, float *data, float *bias, size_t batch_size,
                                 size_t channel_in, size_t height_in, size_t width_in);

void (*QuantizedConvOpFreeRT)(QuantizedConvOp *p);

QuantizedFCOp *(*QuantizedFCOpCreateRT)();

void (*QuantizedFCOpSetupFCParameterRT)(QuantizedFCOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                        FC_ALGORITHM algo);

void (*QuantizedFCOpInitWeightRT)(QuantizedFCOp *p, float *weight);

void (*QuantizedFCOpExecuteRT)(QuantizedFCOp *p, float *dst, float *data, float *bias, size_t batch_size,
                               size_t channel_in);

void (*QuantizedFCOpFreeRT)(QuantizedFCOp *p);


void BindSymbol() {
#if defined(WINDOWS)
#define BINDSYMBOL GetProcAddress
#else
#define BINDSYMBOL dlsym
#endif
  QuantizedConvOpCreateRT =
      reinterpret_cast<QuantizedConvOp *(*)()>(BINDSYMBOL(handler, "InternalQuantizedConvOpCreate"));
  QuantizedConvOpSetupConvParameterRT =
      reinterpret_cast<void (*)(QuantizedConvOp *, LAYOUT, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
                                size_t, size_t, size_t, size_t, size_t, CONV_ALGORITHM)>(
          BINDSYMBOL(handler, "InternalQuantizedConvOpSetupConvParameter"));
  QuantizedConvOpInitWeightRT =
      reinterpret_cast<void (*)(QuantizedConvOp *, float *)>(BINDSYMBOL(handler, "InternalQuantizedConvOpInitWeight"));
  QuantizedConvOpExecuteRT =
      reinterpret_cast<void (*)(QuantizedConvOp *, float *, float *, float *, size_t, size_t, size_t, size_t)>(
          BINDSYMBOL(handler, "InternalQuantizedConvOpExecute"));
  QuantizedConvOpFreeRT =
      reinterpret_cast<void (*)(QuantizedConvOp *)>(BINDSYMBOL(handler, "InternalQuantizedConvOpFree"));
  QuantizedFCOpCreateRT = reinterpret_cast<QuantizedFCOp *(*)()>(BINDSYMBOL(handler, "InternalQuantizedFCOpCreate"));
  QuantizedFCOpSetupFCParameterRT = reinterpret_cast<void (*)(QuantizedFCOp *, LAYOUT, size_t, size_t, FC_ALGORITHM)>(
      BINDSYMBOL(handler, "InternalQuantizedFCOpSetupFCParameter"));
  QuantizedFCOpInitWeightRT =
      reinterpret_cast<void (*)(QuantizedFCOp *, float *)>(BINDSYMBOL(handler, "InternalQuantizedFCOpInitWeight"));
  QuantizedFCOpExecuteRT = reinterpret_cast<void (*)(QuantizedFCOp *, float *, float *, float *, size_t, size_t)>(
      BINDSYMBOL(handler, "InternalQuantizedFCOpExecute"));
  QuantizedFCOpFreeRT = reinterpret_cast<void (*)(QuantizedFCOp *)>(BINDSYMBOL(handler, "InternalQuantizedFCOpFree"));
#undef BINDSYMBOL
}

int ManualRuntimeLoadLib(char *path) {
#if defined(MANUAL_LOAD)
  char lib_path[300];
  strncpy(lib_path, path, 200);
#if defined(WINDOWS)
  char *ext = ".dll";
#elif defined(__APPLE__)
  char *ext = ".dylib";
#else
  const char *ext = ".so";
#endif
  if (handler == NULL) {
    if (cpuid_support_feature(AVX_512)) {
      strncat(lib_path, "/libbigquant_avx512", 100);
    } else if (cpuid_support_feature(AVX2_FMA)) {
      strncat(lib_path, "/libbigquant_avx2", 100);
    } else if (cpuid_support_feature(SSE4_2)) {
      strncat(lib_path, "/libbigquant_sse42", 100);
    } else {
      fprintf(stderr, "Unsupported ISA. Bigquant supports Instruction Set from SSE42 to AVX512.\n");
      return -1;
    }
    strncat(lib_path, ext, 100);
#if defined(WINDOWS)
    handler = LoadLibrary(lib_path);
#else   // WINSOWS
    handler = dlopen(lib_path, RTLD_NOW | RTLD_NODELETE);
#endif  // WINDOWS
    if (handler == NULL) {
      fprintf(stderr, "%s failed to be loaded.\n", lib_path);
      return -2;
    }
  }
  BindSymbol();
#ifndef WINDOWS
  if (handler != NULL) {
    dlclose(handler);
    handler = NULL;
  }
#endif  // WINDOWS
  return 0;
#else   // MANUAL_LOAD
  std::cerr << "Useless Function. Please build with -DMANUAL_LOAD to enable this function." << std::endl;
  return -3;
#endif  // MANUAL_LOAD
}

void __attribute__((constructor)) init_shared_library() {
#ifndef MANUAL_LOAD
  std::string lib_path;
#if defined(WINDOWS)
  std::string ext = ".dll";
#elif defined(__APPLE__)
  std::string ext = ".dylib";
#else
  std::string ext = ".so";
#endif
  if (handler == NULL) {
    if (cpuid_support_feature(AVX_512)) {
      lib_path = "libbigquant_avx512";
    } else if (cpuid_support_feature(AVX2_FMA)) {
      lib_path = "libbigquant_avx2";
    } else if (cpuid_support_feature(SSE4_2)) {
      lib_path = "libbigquant_sse42";
    } else {
      std::cerr << "Unsupported ISA. Bigquant supports Instruction Set from SSE42 to AVX512.\n" << std::endl;
      exit(-1);
    }
    lib_path += ext;
#if defined(WINDOWS)
    handler = LoadLibrary(lib_path.c_str());
#else   // WINSOWS
    handler = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_NODELETE);
#endif  // WINDOWS
    if (handler == NULL) {
      std::cerr << lib_path.c_str() << " failed to be loaded." << std::endl;
      exit(-1);
    }
  }
  BindSymbol();
#ifndef WINDOWS
  if (handler != NULL) {
    dlclose(handler);
    handler = NULL;
  }
#endif  // WINDOWS
#endif
}

void __attribute__((destructor)) free_shared_library() {
#ifndef MANUAL_LOAD
  if (handler != NULL) {
#if defined(WINDOWS)
    FreeLibrary(handler);
#else   // WINDOWS
    dlclose(handler);
#endif  // WINDOWS
  }
#endif
}

QuantizedConvOp *QuantizedConvOpCreate() {
  return QuantizedConvOpCreateRT();
}

void QuantizedConvOpSetupConvParameter(QuantizedConvOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                       size_t group, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
                                       size_t pad_h, size_t pad_w, size_t dialation_h, size_t dialation_w,
                                       size_t fusion_mask, CONV_ALGORITHM algo) {
  QuantizedConvOpSetupConvParameterRT(p, layout, channel_out, channel_in, group, kernel_h, kernel_w, stride_h, stride_w,
                                      pad_h, pad_w, dialation_h, dialation_w, fusion_mask, algo);
}

void QuantizedConvOpInitWeight(QuantizedConvOp *p, float *weight) {
  QuantizedConvOpInitWeightRT(p, weight);
}

void QuantizedConvOpExecute(QuantizedConvOp *p, float *dst, float *data, float *bias, size_t batch_size,
                            size_t channel_in, size_t height_in, size_t width_in) {
  QuantizedConvOpExecuteRT(p, dst, data, bias, batch_size, channel_in, height_in, width_in);
}

void QuantizedConvOpFree(QuantizedConvOp *p) {
  QuantizedConvOpFreeRT(p);
}

QuantizedFCOp *QuantizedFCOpCreate() {
  return QuantizedFCOpCreateRT();
}

void QuantizedFCOpSetupFCParameter(QuantizedFCOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                   FC_ALGORITHM algo) {
  QuantizedFCOpSetupFCParameterRT(p, layout, channel_out, channel_in, algo);
}

void QuantizedFCOpInitWeight(QuantizedFCOp *p, float *weight) {
  QuantizedFCOpInitWeightRT(p, weight);
}

void QuantizedFCOpExecute(QuantizedFCOp *p, float *dst, float *data, float *bias, size_t batch_size,
                          size_t channel_in) {
  QuantizedFCOpExecuteRT(p, dst, data, bias, batch_size, channel_in);
}

void QuantizedFCOpFree(QuantizedFCOp *p) {
  QuantizedFCOpFreeRT(p);
}

