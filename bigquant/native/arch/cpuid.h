#ifndef ARCH_CPUID_H
#define ARCH_CPUID_H
#include <string>
#include "../common.h"

static bool cpuid_support_feature(CPU_FEATURE f) {
  bool support_sse4_2;
  bool support_avx2;
  bool support_fma;
  bool support_avx512;
  {
    uint32_t eax, ebx, ecx, edx;
    eax = 1;
    __asm__("cpuid" : "+a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx));
    support_sse4_2 = ecx & (1 << 20);
    support_fma = ecx & (1 << 12);
  }
  {
    uint32_t eax, ebx, ecx, edx;
    eax = 7;
    ecx = 0;
    __asm__("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));
    support_avx2 = ebx & (1 << 5);
    support_avx512 = ebx & ((1 << 16) + (1 << 17) + (1 << 30) + (1 << 31));
  }
  if (f == SSE4_2) {
    return support_sse4_2;
  } else if (f == AVX2_FMA) {
    return support_fma && support_avx2;
  } else if (f == AVX_512) {
    return support_avx512;
  } else {
    throw "Unknown CPU ISA. Internal Error.\n";
  }
}

struct cpuinfo {
  size_t l1_cache_size_per_core;
  size_t l2_cache_size_per_core;
  size_t l1_cache_size_per_active_thread;
  size_t l2_cache_size_per_active_thread;
  size_t l3_cache_size_per_package;
  bool l2_include_l1;
  bool l3_include_l2;
  size_t max_logic_core_num_per_package; //  logic core num when ht on
  size_t active_logic_core_num_per_package; //  logic core num when ht on
  size_t max_logic_core_num;
  size_t active_logic_core_num;
  size_t package_num;
  bool hyperthreading_on;
};

bool is_hyperthreading_on() {
#if defined(WINDOWS)
  return true;
#else
  std::mutex mutex_thread_lock;
  std::vector<std::thread> thread_groups(std::thread::hardware_concurrency());
  size_t even_processor_num = 0;
  size_t odd_processor_num = 0;
  for (size_t t = 0; t < std::thread::hardware_concurrency(); ++t) {
    cpu_set_t affinity_set;
    CPU_ZERO(&affinity_set);
    CPU_SET(t, &affinity_set);
    mutex_thread_lock.lock();
    thread_groups[t] = std::thread([&mutex_thread_lock, t, &even_processor_num, &odd_processor_num]() {
      mutex_thread_lock.lock();
      uint32_t eax, ebx, ecx, edx;
      eax = 0xb;
      ecx = 1;
      __asm__("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));
      if ((edx & 1) == 0) {
        ++odd_processor_num;
      } else {
        ++even_processor_num;
      }
      mutex_thread_lock.unlock();
    }
    );
    pthread_setaffinity_np(thread_groups[t].native_handle(), sizeof(cpu_set_t), &affinity_set);
    mutex_thread_lock.unlock();
  }

  for (size_t t = 0; t < std::thread::hardware_concurrency(); ++t) {
    thread_groups[t].join();
  }
#if defined(DEBUG)
  std::cerr << "even_processor_num " << even_processor_num << std::endl;
  std::cerr << "odd_processor_num " << odd_processor_num << std::endl;
#endif
  return even_processor_num == odd_processor_num;
#endif
}


struct cpuinfo get_cpuinfo() {
  struct cpuinfo info;
  uint32_t eax, ebx, ecx, edx;
  info.active_logic_core_num = std::thread::hardware_concurrency();
  info.hyperthreading_on = is_hyperthreading_on();
  info.max_logic_core_num = (info.hyperthreading_on)? info.active_logic_core_num : info.active_logic_core_num * 2;
  {
    eax = 4;  // get cache info
    ecx = 0;  // L1 Data Cache

    __asm__("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));

    int cache_type = eax & 31;

    if (cache_type == 0) {
      info.l1_cache_size_per_core = -1;
    } else {
      bool inclusive = (edx >> 1) & 1;
      size_t cache_level = (eax >> 5) & 0x7;
      size_t cache_sets = ecx + 1;
      size_t cacheline_size = (ebx & 0xfff) + 1;
      size_t cacheline_partitions = ((ebx >> 12) & 0x3ff) + 1;
      size_t cache_ways = ((ebx >> 22) & 0x3ff) + 1;
      size_t cache_size = cache_ways * cacheline_partitions * cacheline_size * cache_sets;
      info.l1_cache_size_per_core = cache_size;
      info.l1_cache_size_per_active_thread = (info.hyperthreading_on)? cache_size / 2 : cache_size;
    }
  }

  {
    eax = 4;  // get cache info
    ecx = 2;  // L2 Data Cache

    __asm__("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));

    int cache_type = eax & 31;

    if (cache_type == 0) {
      info.l2_cache_size_per_core = -1;
    } else {
      bool inclusive = (edx >> 1) & 1;
      size_t cache_level = (eax >> 5) & 0x7;
      size_t cache_sets = ecx + 1;
      size_t cacheline_size = (ebx & 0xfff) + 1;
      size_t cacheline_partitions = ((ebx >> 12) & 0x3ff) + 1;
      size_t cache_ways = ((ebx >> 22) & 0x3ff) + 1;
      size_t cache_size = cache_ways * cacheline_partitions * cacheline_size * cache_sets;
      info.l2_cache_size_per_core = cache_size;
      info.l2_cache_size_per_active_thread = (info.hyperthreading_on)? cache_size / 2 : cache_size;
      info.l2_include_l1 = inclusive;
    }
  }

  {
    eax = 4;  // get cache info
    ecx = 3;  // L1 Data Cache

    __asm__("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));

    int cache_type = eax & 31;

    if (cache_type == 0) {
      info.l3_cache_size_per_package = -1;
    } else {
      bool inclusive = (edx >> 1) & 1;
      size_t cache_level = (eax >> 5) & 0x7;
      size_t cache_sets = ecx + 1;
      size_t cacheline_size = (ebx & 0xfff) + 1;
      size_t cacheline_partitions = ((ebx >> 12) & 0x3ff) + 1;
      size_t cache_ways = ((ebx >> 22) & 0x3ff) + 1;
      size_t cache_size = cache_ways * cacheline_partitions * cacheline_size * cache_sets;
      info.l3_cache_size_per_package = cache_size;
      info.l3_include_l2 = inclusive;
    }
  }

  {
    eax = 0xb;
    ecx = 1;
    __asm__("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));

    info.max_logic_core_num_per_package = ebx;
  }
  info.package_num = info.max_logic_core_num / info.max_logic_core_num_per_package;
  info.active_logic_core_num_per_package = info.active_logic_core_num / info.package_num;

#if defined(DEBUG)
  std::cerr << "l1_cache_size_per_core " << info.l1_cache_size_per_core << std::endl;
  std::cerr << "l2_cache_size_per_core " << info.l2_cache_size_per_core << std::endl;
  std::cerr << "l1_cache_size_per_active_thread " << info.l1_cache_size_per_active_thread << std::endl;
  std::cerr << "l2_cache_size_per_active_thread " << info.l2_cache_size_per_active_thread << std::endl;
  std::cerr << "l3_cache_size_per_package " << info.l3_cache_size_per_package << std::endl;
  std::cerr << "l2_include_l1 " << info.l2_include_l1 << std::endl;
  std::cerr << "l3_include_l2 " << info.l3_include_l2 << std::endl;
  std::cerr << "max_logic_core_num_per_package " << info.max_logic_core_num_per_package << std::endl;
  std::cerr << "active_logic_core_num_per_package " << info.active_logic_core_num_per_package << std::endl;
  std::cerr << "max_logic_core_num " << info.max_logic_core_num << std::endl;
  std::cerr << "active_logic_core_num " << info.active_logic_core_num << std::endl;
  std::cerr << "package_num " << info.package_num << std::endl;
  std::cerr << "hyperthreading_on " << info.hyperthreading_on << std::endl;
#endif
  return info;
}

#endif
