// gfx950 instruction wrappers, scheduling rules and lane views. A wrapper named after an instruction emits
// exactly that instruction.
#pragma once

#include "attention_config.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-attributes"    // quiets opus's __host__ __device__ deduction guide
#include "opus/opus.hpp"                                    // vendored aiter opus (MIT), csrc/opus
#pragma clang diagnostic pop
#include <type_traits>

// gfx950 rules, cited by tag:
//   [mfma]    A 16x16x32 MFMA leaves the partner 2 VALU slots; an exp takes both (issued second, it delays the MFMA).
//   [pk]      Packed fp32 (v_pk_*_f32) gets no issue slot beside an MFMA.
//   [salu]    SALU co-issues freely beside exps (one per exp); inside the MFMA block it delays the MFMAs.
//   [branch]  A taken branch costs about 20 cycles of instruction fetch.
//   [vcc]     SALU reading a VALU compare (ballot) waits about 24 cycles for it.
//   [lds]     An LDS read right behind the wave's own exp waits about 16-20 cycles for it.
//   [dma]     A DMA instruction alone behind MFMAs hides its issue stall.
//   [fence]   The MFMA block scheduled as one unit loses its read-ahead: its steps are fenced.
//   [lgkm]    Scalar loads share lgkmcnt with LDS reads and finish out of order: one in flight turns
//             every LDS wait into lgkmcnt(0).

// Kernel lambdas must inline; out of line they spill the accumulators.
#define INLINE __attribute__((always_inline))

namespace attn {

// ---- lane views and notation --------------------------------------------------------------------
// MFMA fragment layout of v_mfma_f32_16x16x32_bf16, lane = 16g + l:
//   A[s] = A[m=l][k=8g+s]    B[s] = B[n=l][k=8g+s]    C[i] = C[m=4g+i][n=l]
// Lane views: a tensor spread across a wave's lanes. Each dim splits into sub-dims, major first: P<n> spans
// lanes (or waves), Y<n> is walked by one lane. "[kv][d] at [a, b][c, 8]": kv splits into a, b and d into c, 8.
// lane_view takes each dim's stride and the lane's P coords; a Y coord y maps to offset .lane + .step(y).
// A [4][32] tile, lane holding row lane/16 and columns lane%16 and 16 + lane%16:
//   auto t = lane_view<xdim<P<4>>, xdim<Y<2>, P<16>>>(make_tuple(row_stride, 1_I), lane / 16, lane % 16);
//   t(1) == (lane / 16) * row_stride + 16 + lane % 16
using opus::operator""_I;
template <int N> struct P { using dim = opus::p_dim; static constexpr int size = N; };
template <int N> struct Y { using dim = opus::y_dim; static constexpr int size = N; };
template <class... SubDims> struct xdim {
    using dims = opus::tuple<typename SubDims::dim...>;
    static constexpr auto shape() { return opus::make_tuple(opus::number<SubDims::size>{}...); }
};

template <int NumY, class Offset, class StepLayout>
struct LaneView {
    Offset lane;
    StepLayout at_lane0;
    template <class... Ys>
    __device__ constexpr auto step(Ys... y) const
    {
        const auto zeros = opus::make_repeated_tuple<NumY - sizeof...(Ys)>(0_I);   // trailing Y coords default to zero
        return at_lane0(opus::concat_tuple(opus::make_tuple(y...), zeros));
    }
    template <class... Ys>
    __device__ constexpr auto operator()(Ys... y) const { return lane + step(y...); }
};

template <class... XDims, class XStride, class... Ps>
__device__ constexpr auto lane_view(XStride x_stride, Ps... p)
{
    constexpr auto dims  = opus::tuple<typename XDims::dims...>{};
    constexpr auto shape = opus::flatten_tuple(opus::make_tuple(XDims::shape()...));
    constexpr int NumY   = opus::size<decltype(shape)>() - sizeof...(Ps);
    const auto stride    = opus::unfold_x_stride(dims, shape, x_stride);
    const auto p_lane    = opus::unfold_p_coord(dims, opus::make_tuple(p...));
    const auto p_zero    = opus::unfold_p_coord(dims, opus::make_repeated_tuple<sizeof...(Ps)>(0_I));
    const auto at_lane   = opus::make_layout<-1>(shape, stride, p_lane);
    const auto at_lane0  = opus::make_layout<-1>(shape, stride, p_zero);
    const auto lane      = at_lane(opus::make_repeated_tuple<NumY>(0_I));
    return LaneView<NumY, decltype(lane), decltype(at_lane0)>{lane, at_lane0};
}

// ---- memory primitives --------------------------------------------------------------------------
using lds_ptr_t        = __attribute__((address_space(3))) void*;
using lds_addr_t       = const __attribute__((address_space(3))) bf16_t*;
using lds_bf16x4_ptr_t = __attribute__((address_space(3))) bf16x4_t*;

// A buffer resource as words: base (48-bit, stride 0), byte range and config (paged K/V patch it in place).
using rsrc_t = __attribute__((ext_vector_type(4))) uint32_t;

// Global-to-LDS DMA, 16 B per lane: lane L writes lds_dst + 16 L. The kernels wait for it themselves (vmcnt,
// then a barrier), not before each LDS read.
template <class Rsrc>
__device__ __forceinline__ void buffer_load_dwordx4_lds(Rsrc rsrc, lds_ptr_t lds_dst, uint32_t voffset,
                                                        uint32_t soffset)
{
    const uint32_t m0 = __builtin_amdgcn_readfirstlane((uint32_t)(uintptr_t)lds_dst);
    // s_nop 0: an SALU M0 write needs one wait state before the LDS DMA reads it.
    asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dwordx4 %1, %2, %3 offen lds"
                 :: "s"(m0), "v"(voffset), "s"(rsrc), "s"(soffset) : "memory");
}

// The same DMA split in two, M0 set ahead (no s_nop): set_m0(lds_dst), then buffer_load_dwordx4_lds_m0 with no
// M0 write between. `after`: the register the load is ordered after.
__device__ __forceinline__ void set_m0(uint32_t lds_dst)
{
    asm volatile("s_mov_b32 m0, %0" :: "s"(__builtin_amdgcn_readfirstlane(lds_dst)));
}

template <class Rsrc, class T>
__device__ __forceinline__ void buffer_load_dwordx4_lds_m0(Rsrc rsrc, uint32_t voffset, uint32_t soffset,
                                                           const T& after)
{
    asm volatile("buffer_load_dwordx4 %0, %1, %2 offen lds" :: "v"(voffset), "s"(rsrc), "s"(soffset),
                 "v"(after) : "memory");
}

// LDS reads; the compiler folds constant offsets and inserts the waits.
__device__ __forceinline__ bf16x8_t ds_read_b128(lds_addr_t p)
{
    return *reinterpret_cast<const __attribute__((address_space(3))) bf16x8_t*>(p);
}

__device__ __forceinline__ bf16x4_t ds_read_b64_tr_b16(lds_addr_t p)
{
    return __builtin_amdgcn_ds_read_tr16_b64_v4bf16((lds_bf16x4_ptr_t)p);
}

// Two fragments joined into one operand of consecutive registers.
__device__ __forceinline__ bf16x8_t join(bf16x4_t a, bf16x4_t b)
{
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7);
}

__device__ __forceinline__ bf16x16_t join(bf16x8_t a, bf16x8_t b)
{
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

// ---- barriers and waits -------------------------------------------------------------------------
// At most N global loads / DMAs in flight. Other waves see a wave's DMA only after this wait and a barrier.
template <int N>
__device__ __forceinline__ void s_waitcnt_vmcnt()
{
    asm volatile("s_waitcnt vmcnt(%0)" :: "n"(N) : "memory");
}

// A bare barrier; the kernels place their own DMA and LDS waits around it.
__device__ __forceinline__ void s_barrier()
{
    asm volatile("s_barrier" ::: "memory");
}

// ---- matrix core ISA ----------------------------------------------------------------------------
__device__ __forceinline__ f32x4_t v_mfma_f32_16x16x32_bf16(bf16x8_t a, bf16x8_t b, f32x4_t c)
{
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
}

// Sparse MFMA, used solely as a row sum: a is 2:4-compressed, idx its per-lane selector.
__device__ __forceinline__ f32x4_t v_smfmac_f32_16x16x64_bf16(bf16x8_t a, bf16x16_t b, f32x4_t c,
                                                              int idx)
{
    return __builtin_amdgcn_smfmac_f32_16x16x64_bf16(a, b, c, idx, 0, 0);
}

// ---- cross-lane and max -------------------------------------------------------------------------
// A single v_max3_f32, which drops a NaN operand.
__device__ __forceinline__ float max3(float a, float b, float c)
{
    return fmaxf(fmaxf(a, b), c);
}

// A single v_maximum3_f32: any NaN operand yields NaN. Operands are MFMA results or maybe_nan values; only a
// may be a constant.
__device__ __forceinline__ float maximum3(float a, float b, float c)
{
    return __builtin_elementwise_maximum(__builtin_elementwise_maximum(a, b), c);
}

// x made opaque to LLVM. -ffast-math then assumes a traceable non-NaN value: a max over such values
// (maximum3 results, constants) lowers to a NaN-dropping v_max, and a NaN test on one folds away.
__device__ __forceinline__ float maybe_nan(float x)
{
    asm("" : "+v"(x));
    return x;
}

// Whether x is set in any lane (wave-uniform).
__device__ __forceinline__ bool any_lane(bool x)
{
    return __builtin_amdgcn_ballot_w64(x) != 0;
}

// Lanes where x > y or x is NaN. A trick compare: -ffast-math lowers !(x <= y) to x > y.
__device__ __forceinline__ uint64_t lanes_above_or_nan(float x, float y)
{
    constexpr int FcmpUgt = 10;                    // llvm CmpInst::FCMP_UGT (unordered-or-greater)
    return __builtin_amdgcn_fcmpf(x, y, FcmpUgt);
}

// Whether lanes holds any of row l's four lanes (l, l + 16, l + 32, l + 48).
__device__ __forceinline__ bool has_row(uint64_t lanes, int l)
{
    const uint32_t rows = (uint32_t)(lanes | lanes >> 32);
    return (rows | rows >> 16) >> l & 1;
}

// Whether x's bits are NaN or +inf. x must come from maybe_nan: otherwise LLVM folds the test (v_cmp_class
// too: under -ffast-math clang marks a float argument neither NaN nor Inf).
__device__ __forceinline__ bool nan_or_plus_inf(float x)
{
    const uint32_t bits = __builtin_bit_cast(uint32_t, x);
    return bits == 0x7F800000u || (bits & 0x7FFFFFFFu) > 0x7F800000u;
}

// A NaN or -inf hidden from the compiler, which -ffast-math assumes cannot occur.
__device__ __forceinline__ float opaque_float(uint32_t bits)
{
    float x = __builtin_bit_cast(float, bits);
    asm volatile("" : "+v"(x));
    return x;
}
__device__ __forceinline__ float opaque_nan() { return opaque_float(0x7FC00000u); }
__device__ __forceinline__ float opaque_neg_inf() { return opaque_float(0xFF800000u); }

// x[i] = max(x[i], x[i] of lane ^ 32), then lane ^ 16. Early-clobber copies keep the self-swap from
// folding; issued first, they cover the VALU write -> permlane read hazard without an s_nop.
__device__ __forceinline__ void max_lane_xor32(float (&x)[3])
{
    float a = x[0], b = x[1], c = x[2], ta, tb, tc;
    asm volatile("v_mov_b32 %3, %0\n\tv_mov_b32 %4, %1\n\tv_mov_b32 %5, %2\n\t"
                 "v_permlane32_swap_b32 %0, %3 bound_ctrl:1\n\t"
                 "v_permlane32_swap_b32 %1, %4 bound_ctrl:1\n\t"
                 "v_permlane32_swap_b32 %2, %5 bound_ctrl:1"
                 : "+v"(a), "+v"(b), "+v"(c), "=&v"(ta), "=&v"(tb), "=&v"(tc));
    x[0] = fmaxf(a, ta), x[1] = fmaxf(b, tb), x[2] = fmaxf(c, tc);
}

__device__ __forceinline__ void max_lane_xor16(float (&x)[3])
{
    float a = x[0], b = x[1], c = x[2], ta, tb, tc;
    asm volatile("v_mov_b32 %3, %0\n\tv_mov_b32 %4, %1\n\tv_mov_b32 %5, %2\n\t"
                 "v_permlane16_swap_b32 %0, %3 bound_ctrl:1\n\t"
                 "v_permlane16_swap_b32 %1, %4 bound_ctrl:1\n\t"
                 "v_permlane16_swap_b32 %2, %5 bound_ctrl:1"
                 : "+v"(a), "+v"(b), "+v"(c), "=&v"(ta), "=&v"(tb), "=&v"(tc));
    x[0] = fmaxf(a, ta), x[1] = fmaxf(b, tb), x[2] = fmaxf(c, tc);
}

// Two values (QTiles 2) padded up to the 3-value asm.
__device__ __forceinline__ void max_lane_xor32(float (&x)[2])
{
    float y[3] = {x[0], x[1], x[0]};
    max_lane_xor32(y);
    x[0] = y[0], x[1] = y[1];
}
__device__ __forceinline__ void max_lane_xor16(float (&x)[2])
{
    float y[3] = {x[0], x[1], x[0]};
    max_lane_xor16(y);
    x[0] = y[0], x[1] = y[1];
}
// The max over x, NaN if any x is NaN (maximum3's rules; with two values, MaskScore fills a).
__device__ __forceinline__ float maximum_n(const float (&x)[3]) { return maximum3(x[0], x[1], x[2]); }
__device__ __forceinline__ float maximum_n(const float (&x)[2]) { return maximum3(MaskScore, x[0], x[1]); }

// A wave-uniform float in an SGPR (the builtin folds to a VGPR copy). s_nop 1 covers the VALU write ->
// readfirstlane hazard; LLVM adds no wait states before asm.
__device__ __forceinline__ float readfirstlane(float x)
{
    float s;
    asm("s_nop 1\n\tv_readfirstlane_b32 %0, %1" : "=s"(s) : "v"(x));
    return s;
}

// ---- instruction scheduling ---------------------------------------------------------------------
// __builtin_amdgcn_sched_group_barrier's instruction classes.
constexpr int SchedValu = 0x002, SchedSalu = 0x004, SchedMfma = 0x008, SchedDsRead = 0x100,
              SchedTrans = 0x400;

// No instruction may move across this point.
__device__ __forceinline__ void sched_fence()
{
    __builtin_amdgcn_sched_barrier(0);
}

// A rare condition: its code goes out of line. A macro, because inside a function __builtin_expect is lost at
// inlining.
#define unlikely(x) __builtin_expect(!!(x), 0)

// Empty-asm data edges (MFMAs move across asm barriers): pin_before(x) keeps x's producers above,
// pin_after(x) its consumers below; no MFMA hazard wait.
template <class T>
__device__ __forceinline__ void pin_before(const T& x)
{
    asm volatile("" :: "v"(x));
}

template <class T, int N, int M>
__device__ __forceinline__ void pin_before(const T (&x)[N][M])
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j) pin_before(x[i][j]);
}

template <class T>
__device__ __forceinline__ void pin_after(T& x)
{
    asm volatile("" : "+v"(x));
}

// x is in an SGPR here: its scalar load has landed [lgkm].
template <class T>
__device__ __forceinline__ void pin_scalar(const T& x)
{
    asm volatile("" :: "s"(x));
}

// x is an SGPR redefined here: its users stay below, its producers above.
template <class T>
__device__ __forceinline__ void anchor(T& x)
{
    asm volatile("" : "+s"(x));
}

// This wave's issue priority over the other wave sharing its SIMD.
template <int P>
__device__ __forceinline__ void s_setprio()
{
    asm volatile("s_setprio %0" :: "n"(P) : "memory");
}

// The lane id, recomputed at each use so no lane-derived value stays live across the loop.
__device__ __forceinline__ int fresh_lane_id()
{
    int id;
    asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\tv_mbcnt_hi_u32_b32 %0, -1, %0" : "=v"(id));
    return id;
}

}  // namespace attn
