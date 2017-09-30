#ifndef COMMON_H
#define COMMON_H
#include "base.h"
#include "arch/config.h"
#include "arch/cpuid.h"
#ifdef NUMA
#include <numa.h>
#endif
#include "alloc.h"

static INLINE_SPECIFIER bool x_ge_0_and_x_lt_bound(int x, int bound) {
  return (0 <= x) && (x < bound);
}

INLINE_SPECIFIER size_t GetConvOutSize(size_t in, size_t kernel, size_t stride, size_t pad, size_t dilation) {
  return (in + 2 * pad - (dilation * (kernel - 1) + 1)) / stride + 1;
}

INLINE_SPECIFIER size_t GetAlignmentLength(size_t n, size_t alignment = 1) {
  return alignment * static_cast<size_t>(ceil(1.0 * n / alignment));
}

template <typename DType>
void ComputeMatrixSumPerRow(DType *dst, DType *src, size_t m, size_t n) {
#pragma omp parallel for
  for (size_t i = 0; i < m; ++i) {
    DType sum = 0;
    for (size_t j = 0; j < n; ++j) {
      sum += *(src + i * n + j);
    }
    dst[i] = sum;
  }
}


size_t GetBlockSize(size_t x, size_t y) {
  return x * y;
}

size_t GetBlockNum(size_t buffer_size, size_t tile_size, float ratio = 0.5) {
  return std::max(static_cast<size_t>(ratio * buffer_size / tile_size), static_cast<size_t>(1));
}


// TODO(yan): still need some improvement, cannot detect cache relation, unified or private
template <size_t tile_m>
INLINE_SPECIFIER void GetBlocksInfo(size_t m, size_t k, size_t &m_in_l1, size_t &m_in_l2, size_t &m_in_l3) {
  // init static struct
  // init static is threadsaft in C++11
  static struct cpuinfo info = get_cpuinfo();
  size_t block_size = GetBlockSize(tile_m, k);

  size_t l1_cache_size = info.l1_cache_size_per_active_thread;
  size_t block_num_per_L1 = GetBlockNum(l1_cache_size, block_size);

  size_t l2_cache_size = info.l2_cache_size_per_active_thread;
  size_t block_num_per_L2 = GetBlockNum(l2_cache_size, block_size) / block_num_per_L1 * block_num_per_L1;

  size_t l3_cache_size = info.l3_cache_size_per_package;
  bool has_l3 = l3_cache_size > 0;


#if defined(LLC_PER_CORE)
  l3_cache_size /= info.active_logic_core_num_per_package;
  if (info.l3_include_l2 == false) {
    l3_cache_size += info.l2_cache_size_per_active_thread;
  }
#endif
  size_t block_num_per_L3 = GetBlockNum(l3_cache_size, block_size) / block_num_per_L2 * block_num_per_L2;

#if defined(LLC_PER_CORE)
  if (has_l3 == false) {
    m_in_l2 = std::max(std::min(block_num_per_L2 * tile_m, m / info.active_logic_core_num / tile_m * tile_m), tile_m);
    m_in_l1 = std::max(std::min(block_num_per_L1 * tile_m, m_in_l2 / 2 / tile_m * tile_m), tile_m);
    m_in_l3 = m_in_l2;
  } else {
    if (m >= 2 * block_num_per_L3 * tile_m) {
      m_in_l3 = block_num_per_L3 * tile_m;
      m_in_l2 = block_num_per_L2 * tile_m;
      m_in_l1 = block_num_per_L1 * tile_m;
    } else if (m >= 2 * block_num_per_L2 * tile_m) {
      m_in_l3 = m;
      m_in_l2 = block_num_per_L2 * tile_m;
      m_in_l1 = block_num_per_L1 * tile_m;
    } else if (m >= 2 * block_num_per_L1 * tile_m) {
      m_in_l3 = m;
      m_in_l2 = m;
      m_in_l1 = block_num_per_L1 * tile_m;
    } else {
      m_in_l3 = m;
      m_in_l2 = m;
      m_in_l1 = m;
    }
  }
#endif

#if defined(LLC_SHARED)
  m_in_l3 = std::max((has_l3 == false) ? m : std::min(block_num_per_L3 * tile_m, m), tile_m);
  m_in_l2 = std::max(std::min(block_num_per_L2 * tile_m, m_in_l3 / info.active_logic_core_num / tile_m * tile_m), tile_m);
  m_in_l1 = std::max(std::min(block_num_per_L1 * tile_m, m_in_l2 / 2 / tile_m * tile_m), tile_m);
#endif

#if defined(DEBUG)
  std::cerr << "l3:" << l3_cache_size << " l2: " << l2_cache_size << " l1:" << l1_cache_size << std::endl;
  std::cerr << "m:" << m << " m_in_l3:" << m_in_l3 << " m_in_l2: " << m_in_l2 << " m_in_l1:" << m_in_l1 << std::endl;
#endif
}
#endif
