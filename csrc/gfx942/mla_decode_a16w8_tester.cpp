// Standalone correctness and performance tester for mla_decode_a16w8.hip.
//
// Math (q_len == 1, latent_dim == 512, rope_dim == 64):
//   score[b,h,t] = scale * kv_scale *
//                    (dot(q_lat[b,h], kv[b,t,0:512]) +
//                     dot(q_pe[b,h],  kv[b,t,512:576]))
//   out[b,h] = kv_scale * sum_t softmax(score[b,h])[t] * kv[b,t,0:512]
//
// Usage:
//   ./build/mla_decode_a16w8_tester [B] [H] [S] [warmup] [iters] [--force-check]
//   Defaults: B=2 H=16 S=16384 warmup=10 iters=30
//   With only --force-check: B=1 H=16 S=256 warmup=2 iters=5

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

extern "C" {
size_t mla_decode_a16w8_workspace_bytes(void);
int launch_mla_decode_a16w8(
    const void* q_lat, const void* q_pe, const void* c_kv, const void* k_pe, void* o_lat,
    int B, int H, int S, int lat_dim, int rope_dim, float scale, float kv_scale, float scale_p,
    void* workspace, void* stream);
int mla_decode_a16w8_plan_parts(int B, int H, int max_seq_len, int lat_dim);
}

#define HIP_CHECK(call)                                                                    \
  do {                                                                                     \
    const hipError_t error = (call);                                                       \
    if (error != hipSuccess) {                                                             \
      std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,               \
                   hipGetErrorString(error));                                              \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

#define KERNEL_CHECK(call)                                                                 \
  do {                                                                                     \
    const int status = (call);                                                             \
    if (status != 0) {                                                                     \
      std::fprintf(stderr, "Kernel launch failed at %s:%d: status %d\n",                  \
                   __FILE__, __LINE__, status);                                            \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

namespace {

constexpr int kLat = 512;
constexpr int kRope = 64;
constexpr int kFused = kLat + kRope;
constexpr int kRotatingBuffers = 4;

std::string human_bytes(size_t bytes) {
  char text[32];
  if (bytes >= (1ULL << 30))
    std::snprintf(text, sizeof(text), "%.2f GiB", bytes / static_cast<double>(1ULL << 30));
  else if (bytes >= (1ULL << 20))
    std::snprintf(text, sizeof(text), "%.2f MiB", bytes / static_cast<double>(1ULL << 20));
  else if (bytes >= (1ULL << 10))
    std::snprintf(text, sizeof(text), "%.2f KiB", bytes / static_cast<double>(1ULL << 10));
  else
    std::snprintf(text, sizeof(text), "%zu B", bytes);
  return text;
}

bool env_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && std::strcmp(value, "1") == 0;
}

// AMD E4M3-FNUZ: exponent bias 8, one NaN encoding (0x80), no negative zero.
float fp8e4m3fnuz_to_float(uint8_t bits) {
  if (bits == 0x80) {
    const uint32_t qnan = 0x7fc00000u;
    return __builtin_bit_cast(float, qnan);
  }
  const bool negative = (bits & 0x80u) != 0;
  const int exponent = (bits >> 3) & 0xf;
  const int mantissa = bits & 0x7;
  const float magnitude = exponent == 0
      ? std::ldexp(static_cast<float>(mantissa) / 8.0f, -7)
      : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 8);
  return negative ? -magnitude : magnitude;
}

uint8_t float_to_fp8e4m3fnuz(float value) {
  if (value == 0.0f) return 0;
  const uint8_t sign = std::signbit(value) ? 0x80u : 0u;
  float magnitude = std::fabs(value);
  const uint32_t magnitude_bits = __builtin_bit_cast(uint32_t, magnitude);
  if ((magnitude_bits & 0x7f800000u) == 0x7f800000u || magnitude >= 240.0f)
    return sign | 0x7fu;

  constexpr float kMinNormal = 0.0078125f;  // 2^-7
  if (magnitude < kMinNormal) {
    int mantissa = static_cast<int>(std::lround(magnitude * 1024.0f));
    if (mantissa <= 0) return 0;
    if (mantissa >= 8) return sign | 0x08u;
    return sign | static_cast<uint8_t>(mantissa);
  }

  int exponent2 = 0;
  float fraction = std::frexp(magnitude, &exponent2);  // [0.5, 1)
  fraction *= 2.0f;
  --exponent2;
  int exponent = exponent2 + 8;
  int mantissa = static_cast<int>(std::lround((fraction - 1.0f) * 8.0f));
  if (mantissa == 8) {
    mantissa = 0;
    ++exponent;
  }
  if (exponent > 15) return sign | 0x7fu;
  return sign | static_cast<uint8_t>((exponent << 3) | mantissa);
}

bool fp8_self_test() {
  struct TestCase { uint8_t bits; float value; };
  constexpr TestCase cases[] = {
      {0x00, 0.0f}, {0x01, 0.0009765625f}, {0x07, 0.0068359375f},
      {0x08, 0.0078125f}, {0x38, 0.5f}, {0x40, 1.0f},
      {0x41, 1.125f}, {0x48, 2.0f}, {0x78, 128.0f}, {0x7f, 240.0f},
  };
  for (const auto& test : cases) {
    const float decoded = fp8e4m3fnuz_to_float(test.bits);
    if (decoded != test.value || float_to_fp8e4m3fnuz(test.value) != test.bits) {
      std::fprintf(stderr,
                   "FP8 self-test failed for 0x%02x: decoded %.9g, expected %.9g\n",
                   test.bits, decoded, test.value);
      return false;
    }
  }
  return true;
}

void fill_random_bf16(std::vector<__hip_bfloat16>& values, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> distribution(0.0f, 0.5f);
  for (auto& value : values) value = __float2bfloat16(distribution(rng));
}

void fill_random_fp8(std::vector<uint8_t>& values, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> distribution(0.0f, 0.5f);
  for (auto& value : values) value = float_to_fp8e4m3fnuz(distribution(rng));
}

void mla_reference(const std::vector<__hip_bfloat16>& q_lat,
                   const std::vector<__hip_bfloat16>& q_pe,
                   const std::vector<uint8_t>& kv,
                   std::vector<float>& output,
                   int B, int H, int S, float scale, float kv_scale) {
  std::vector<float> scores(static_cast<size_t>(S));
  for (int b = 0; b < B; ++b) {
    const uint8_t* batch_kv = kv.data() + static_cast<size_t>(b) * S * kFused;
    for (int h = 0; h < H; ++h) {
      const __hip_bfloat16* ql = q_lat.data() + (static_cast<size_t>(b) * H + h) * kLat;
      const __hip_bfloat16* qp = q_pe.data() + (static_cast<size_t>(b) * H + h) * kRope;
      float* out = output.data() + (static_cast<size_t>(b) * H + h) * kLat;

      float row_max = std::numeric_limits<float>::lowest();
      for (int token = 0; token < S; ++token) {
        const uint8_t* row = batch_kv + static_cast<size_t>(token) * kFused;
        float dot = 0.0f;
        for (int dim = 0; dim < kLat; ++dim)
          dot += static_cast<float>(ql[dim]) * fp8e4m3fnuz_to_float(row[dim]);
        for (int dim = 0; dim < kRope; ++dim)
          dot += static_cast<float>(qp[dim]) * fp8e4m3fnuz_to_float(row[kLat + dim]);
        scores[token] = dot * scale * kv_scale;
        row_max = std::max(row_max, scores[token]);
      }

      float denominator = 0.0f;
      for (float& score : scores) {
        score = std::exp(score - row_max);
        denominator += score;
      }

      std::fill(out, out + kLat, 0.0f);
      const float inv_denominator = 1.0f / denominator;
      for (int token = 0; token < S; ++token) {
        const uint8_t* row = batch_kv + static_cast<size_t>(token) * kFused;
        const float probability = scores[token] * inv_denominator;
        for (int dim = 0; dim < kLat; ++dim)
          out[dim] += probability * fp8e4m3fnuz_to_float(row[dim]) * kv_scale;
      }
    }
  }
}

struct CorrectnessResult {
  bool passed = false;
  double max_abs = 0.0;
  double max_rel = 0.0;
  double relative_l2 = 0.0;
  size_t mismatches = 0;
};

CorrectnessResult compare_outputs(const std::vector<__hip_bfloat16>& actual,
                                  const std::vector<float>& expected) {
  CorrectnessResult result;
  double squared_error = 0.0;
  double squared_reference = 0.0;
  for (size_t index = 0; index < expected.size(); ++index) {
    const double got = static_cast<float>(actual[index]);
    const double want = expected[index];
    uint16_t bits = 0;
    std::memcpy(&bits, &actual[index], sizeof(bits));
    const bool non_finite = (bits & 0x7f80u) == 0x7f80u;
    const double abs_error = std::fabs(got - want);
    const double rel_error = abs_error / (std::fabs(want) + 1e-6);
    result.max_abs = std::max(result.max_abs, abs_error);
    result.max_rel = std::max(result.max_rel, rel_error);
    squared_error += abs_error * abs_error;
    squared_reference += want * want;
    if (non_finite || (abs_error > 1e-2 && rel_error > 2e-2)) {
      if (result.mismatches < 10) {
        std::fprintf(stderr,
                     "Mismatch at %zu: device=%.6f reference=%.6f abs=%.6f rel=%.6f\n",
                     index, got, want, abs_error, rel_error);
      }
      ++result.mismatches;
    }
  }
  result.relative_l2 = squared_reference > 0.0
      ? std::sqrt(squared_error / squared_reference)
      : std::sqrt(squared_error);
  constexpr double kRelativeL2Tolerance = 2e-2;
  result.passed = result.mismatches == 0 && result.relative_l2 <= kRelativeL2Tolerance;
  return result;
}

struct BenchmarkResult {
  float best_ms = 0.0f;
  float average_ms = 0.0f;
  double best_tflops = 0.0;
  double average_tflops = 0.0;
  bool correctness_checked = false;
  bool correctness_passed = true;
};

BenchmarkResult run_benchmark(int B, int H, int S, int warmup, int iterations,
                              bool check_correctness) {
  const size_t q_lat_count = static_cast<size_t>(B) * H * kLat;
  const size_t q_pe_count = static_cast<size_t>(B) * H * kRope;
  const size_t kv_count = static_cast<size_t>(B) * S * kFused;
  const size_t output_count = q_lat_count;
  const size_t q_lat_bytes = q_lat_count * sizeof(__hip_bfloat16);
  const size_t q_pe_bytes = q_pe_count * sizeof(__hip_bfloat16);
  const size_t kv_bytes = kv_count;
  const size_t output_bytes = output_count * sizeof(__hip_bfloat16);
  const size_t workspace_bytes = mla_decode_a16w8_workspace_bytes();
  const bool no_workspace = env_enabled("MLA_NO_WORKSPACE") ||
                            env_enabled("MOONMATH_NO_TAIL");
  const int parts = no_workspace ? 1 : mla_decode_a16w8_plan_parts(B, H, S, kLat);

  std::printf("  [mem] Q_lat/Q_pe: %d buffers x %s/%s\n", kRotatingBuffers,
              human_bytes(q_lat_bytes).c_str(), human_bytes(q_pe_bytes).c_str());
  std::printf("  [mem] fused KV   : %d buffers x %s; output: %s\n", kRotatingBuffers,
              human_bytes(kv_bytes).c_str(), human_bytes(output_bytes).c_str());
  if (no_workspace)
    std::printf("  [mem] workspace  : disabled; single-part path\n");
  else
    std::printf("  [mem] workspace  : %s; planned parts: %d\n",
                human_bytes(workspace_bytes).c_str(), parts);

  __hip_bfloat16* d_q_lat[kRotatingBuffers]{};
  __hip_bfloat16* d_q_pe[kRotatingBuffers]{};
  uint8_t* d_kv[kRotatingBuffers]{};
  __hip_bfloat16* d_output = nullptr;
  void* d_workspace = nullptr;

  std::printf("  [mem] allocating GPU buffers...\n");
  for (int index = 0; index < kRotatingBuffers; ++index) {
    HIP_CHECK(hipMalloc(&d_q_lat[index], q_lat_bytes));
    HIP_CHECK(hipMalloc(&d_q_pe[index], q_pe_bytes));
    HIP_CHECK(hipMalloc(&d_kv[index], kv_bytes));
  }
  HIP_CHECK(hipMalloc(&d_output, output_bytes));
  if (!no_workspace) HIP_CHECK(hipMalloc(&d_workspace, workspace_bytes));

  std::vector<__hip_bfloat16> h_q_lat(q_lat_count);
  std::vector<__hip_bfloat16> h_q_pe(q_pe_count);
  std::vector<uint8_t> h_kv(kv_count);
  for (int index = 0; index < kRotatingBuffers; ++index) {
    fill_random_bf16(h_q_lat, 42u + index);
    fill_random_bf16(h_q_pe, 100u + index);
    fill_random_fp8(h_kv, 200u + index);
    HIP_CHECK(hipMemcpy(d_q_lat[index], h_q_lat.data(), q_lat_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_q_pe[index], h_q_pe.data(), q_pe_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_kv[index], h_kv.data(), kv_bytes, hipMemcpyHostToDevice));
    std::printf("  [data] buffer %d/%d uploaded\n", index + 1, kRotatingBuffers);
  }

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  const float scale = 1.0f / std::sqrt(static_cast<float>(kFused));
  const float kv_scale = 1.0f;
  const float scale_p = 1.0f;  // ABI compatibility; unused by this kernel.
  auto launch = [&](int index) {
    KERNEL_CHECK(launch_mla_decode_a16w8(
        d_q_lat[index], d_q_pe[index], d_kv[index], nullptr, d_output,
        B, H, S, kLat, kRope, scale, kv_scale, scale_p,
        no_workspace ? nullptr : d_workspace, reinterpret_cast<void*>(stream)));
  };

  std::printf("\n  [phase] WARMUP (%d iterations) ", warmup);
  std::fflush(stdout);
  for (int iteration = 0; iteration < warmup; ++iteration) {
    launch(iteration % kRotatingBuffers);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::printf(".");
    std::fflush(stdout);
  }
  std::printf(" done\n\n");

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  std::vector<float> times;
  times.reserve(iterations);
  int last_index = 0;
  std::printf("  [phase] TIMING (%d iterations)\n", iterations);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    last_index = iteration % kRotatingBuffers;
    HIP_CHECK(hipEventRecord(start, stream));
    launch(last_index);
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    times.push_back(elapsed_ms);
    const int log_step = iterations <= 10 ? 1 : std::max(1, iterations / 5);
    if (iteration == 0 || (iteration + 1) % log_step == 0)
      std::printf("  iter %3d: %7.3f ms\n", iteration + 1, elapsed_ms);
  }
  std::printf("  [phase] TIMING done\n\n");

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  const float best_ms = *std::min_element(times.begin(), times.end());
  double total_ms = 0.0;
  for (float time : times) total_ms += time;
  const float average_ms = static_cast<float>(total_ms / times.size());
  // QK: 2*B*H*S*(512+64), PV: 2*B*H*S*512.
  const double flops = 2.0 * B * H * S * (kFused + kLat);

  bool correctness_passed = true;
  if (check_correctness) {
    std::printf("  [phase] CORRECTNESS (CPU reference)\n");
    std::vector<__hip_bfloat16> h_output(output_count);
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, output_bytes, hipMemcpyDeviceToHost));
    fill_random_bf16(h_q_lat, 42u + last_index);
    fill_random_bf16(h_q_pe, 100u + last_index);
    fill_random_fp8(h_kv, 200u + last_index);
    std::vector<float> reference(output_count);
    mla_reference(h_q_lat, h_q_pe, h_kv, reference, B, H, S, scale, kv_scale);
    const CorrectnessResult check = compare_outputs(h_output, reference);
    correctness_passed = check.passed;
    std::printf("  max abs error : %.6f\n", check.max_abs);
    std::printf("  max rel error : %.6f\n", check.max_rel);
    std::printf("  relative L2   : %.6f (tol 0.020)\n", check.relative_l2);
    std::printf("  mismatches    : %zu / %zu\n", check.mismatches, output_count);
    std::printf("  Correctness   : %s\n\n", check.passed ? "PASSED" : "FAILED");
  }

  HIP_CHECK(hipStreamDestroy(stream));
  for (int index = 0; index < kRotatingBuffers; ++index) {
    HIP_CHECK(hipFree(d_q_lat[index]));
    HIP_CHECK(hipFree(d_q_pe[index]));
    HIP_CHECK(hipFree(d_kv[index]));
  }
  HIP_CHECK(hipFree(d_output));
  if (d_workspace) HIP_CHECK(hipFree(d_workspace));

  return {best_ms, average_ms,
          flops / (best_ms * 1e-3) / 1e12,
          flops / (average_ms * 1e-3) / 1e12,
          check_correctness, correctness_passed};
}

void print_usage(const char* program) {
  std::fprintf(stderr,
               "Usage: %s [B] [H] [S] [warmup] [iters] [--force-check]\n",
               program);
}

}  // namespace

int main(int argc, char** argv) {
  int B = 2;
  int H = 16;
  int S = 16384;
  int warmup = 10;
  int iterations = 30;
  bool force_check = false;
  std::vector<std::string> positional;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--force-check") {
      force_check = true;
    } else if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (argument.rfind("--", 0) == 0) {
      std::fprintf(stderr, "Unknown option: %s\n", argument.c_str());
      print_usage(argv[0]);
      return 2;
    } else {
      positional.push_back(argument);
    }
  }
  if (positional.size() > 5) {
    print_usage(argv[0]);
    return 2;
  }
  if (!positional.empty()) B = std::atoi(positional[0].c_str());
  if (positional.size() > 1) H = std::atoi(positional[1].c_str());
  if (positional.size() > 2) S = std::atoi(positional[2].c_str());
  if (positional.size() > 3) warmup = std::atoi(positional[3].c_str());
  if (positional.size() > 4) iterations = std::atoi(positional[4].c_str());
  if (force_check && positional.empty()) {
    B = 1;
    H = 16;
    S = 256;
    warmup = 2;
    iterations = 5;
  }
  if (B <= 0 || H <= 0 || H > 16 || S <= 0 || warmup < 0 || iterations <= 0) {
    std::fprintf(stderr,
                 "Expected B>0, 1<=H<=16, S>0, warmup>=0, and iters>0.\n");
    return 2;
  }
  if (!fp8_self_test()) return 2;

  const uint64_t reference_work = static_cast<uint64_t>(B) * H * S * (kFused + kLat);
  constexpr uint64_t kAutoCheckWork = static_cast<uint64_t>(1) * 16 * 512 * (kFused + kLat);
  const bool check_correctness = force_check || reference_work <= kAutoCheckWork;
  const char* check_mode = force_check ? "ENABLED (--force-check)"
                           : check_correctness ? "ENABLED (auto: small config)"
                                               : "DISABLED (performance only)";

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::fprintf(stderr, "No HIP devices found.\n");
    return 1;
  }
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));

  std::printf("===============================================================\n");
  std::printf("moonmath MLA A16W8 correctness + performance tester\n");
  std::printf("  Device    : %s\n", properties.name);
  std::printf("  Arch      : %s\n", properties.gcnArchName);
  std::printf("  Config    : B=%d H=%d S=%d Lat=512 Rope=64\n", B, H, S);
  std::printf("  Iterations: warmup=%d timed=%d\n", warmup, iterations);
  std::printf("  Check     : %s\n", check_mode);
  std::printf("===============================================================\n\n");

  const BenchmarkResult result =
      run_benchmark(B, H, S, warmup, iterations, check_correctness);

  std::printf("===============================================================\n");
  std::printf("RESULTS\n");
  std::printf("  Best : %8.3f ms | %7.3f TFLOP/s\n", result.best_ms, result.best_tflops);
  std::printf("  Avg  : %8.3f ms | %7.3f TFLOP/s\n", result.average_ms,
              result.average_tflops);
  std::printf("  Check: %s\n", result.correctness_checked
              ? (result.correctness_passed ? "PASSED" : "FAILED") : "SKIPPED");
  std::printf("===============================================================\n");
  return result.correctness_passed ? 0 : 1;
}
