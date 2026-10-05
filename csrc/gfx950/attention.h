// C API for the gfx950 bf16 attention forward: attention.hip (head dim 128) and attention_mla.hip
// (MLA prefill, Q/K head dim 192 and V/O 128). Every call enqueues its kernels on `stream`, returning 0,
// -1 for bad arguments, or a hipError_t.
//
//   tensors  q [.., H, Dqk], k [.., HKv, Dqk], v [.., HKv, Dv], o [.., H, Dv]: either a batch
//            [B, S, heads, d] with B, S, heads in any order, or varlen THD [tokens, heads, d]. H is a
//            multiple of HKv (GQA).
//   strides  A host array in elements, with a contiguous head dim: q's batch, token and head strides,
//            then k's, v's and o's (12 total), then lse's when lse is supplied (15). BSHD reads
//            t.stride(0), t.stride(1), t.stride(2); BHSD reads t.stride(0), t.stride(2), t.stride(1).
//            Varlen disregards batch strides. Tokens lie at most 2^22 elements apart. Entries lacking
//            _strided are contiguous BSHD or THD and take no lse.
//   lse      Either null, or every q row's fp32 natural-log log-sum-exp (-inf for a row with no keys),
//            laid out per its strides: aiter's [B, H, Sq] is H Sq, 1, Sq; varlen [H, total_q] is
//            0, 1, total_q.
//   causal   Aligned bottom-right within every sequence.  scale: softmax_scale, positive and finite.
//   D        Head dim for the head-dim-128 entries, which must equal 128; the MLA entries have none.
#pragma once

#include <hip/hip_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A batch: B sequences, each holding Sq q rows and Skv keys.
int cdna4_attn_fwd_bf16_strided(const void* q, const void* k, const void* v, void* o, float* lse,
                                const int64_t* strides, int B, int H, int HKv, int Sq, int Skv, int D,
                                int causal, float scale, hipStream_t stream);
int cdna4_attn_mla_fwd_bf16_strided(const void* q, const void* k, const void* v, void* o, float* lse,
                                    const int64_t* strides, int B, int H, int HKv, int Sq, int Skv,
                                    int causal, float scale, hipStream_t stream);
int cdna4_attn_fwd_bf16(const void* q, const void* k, const void* v, void* o, int B, int H, int HKv, int Sq,
                        int Skv, int D, int causal, float scale, hipStream_t stream);
int cdna4_attn_mla_fwd_bf16(const void* q, const void* k, const void* v, void* o, int B, int H, int HKv,
                            int Sq, int Skv, int causal, float scale, hipStream_t stream);

// Varlen: sequence i owns Q/O tokens cu_seqlens_q[i] .. cu_seqlens_q[i + 1] - 1 and K/V tokens
// cu_seqlens_k[i] .. cu_seqlens_k[i + 1] - 1; both arrays hold num_seqs + 1 ints in host memory.
int cdna4_attn_varlen_fwd_bf16_strided(const void* q, const void* k, const void* v, void* o, float* lse,
                                       const int64_t* strides, const int* cu_seqlens_q,
                                       const int* cu_seqlens_k, int num_seqs, int H, int HKv, int D,
                                       int causal, float scale, hipStream_t stream);
int cdna4_attn_mla_varlen_fwd_bf16_strided(const void* q, const void* k, const void* v, void* o, float* lse,
                                           const int64_t* strides, const int* cu_seqlens_q,
                                           const int* cu_seqlens_k, int num_seqs, int H, int HKv, int causal,
                                           float scale, hipStream_t stream);
int cdna4_attn_varlen_fwd_bf16(const void* q, const void* k, const void* v, void* o, const int* cu_seqlens_q,
                               const int* cu_seqlens_k, int num_seqs, int H, int HKv, int D, int causal,
                               float scale, hipStream_t stream);
int cdna4_attn_mla_varlen_fwd_bf16(const void* q, const void* k, const void* v, void* o,
                                   const int* cu_seqlens_q, const int* cu_seqlens_k, int num_seqs, int H,
                                   int HKv, int causal, float scale, hipStream_t stream);

// Paged K/V: k and v are token pools broken into pages of page_size tokens (a power of two, at least 8;
// 16 for MLA), with page strides serving as their batch strides (head-dim-128: k and v share strides).
// Sequence i owns Q/O tokens cu_seqlens_q[i] .. cu_seqlens_q[i + 1] - 1 and kv_lens[i] keys, where key r
// is token r % page_size of page page_table[page_starts[i] + r / page_size]. cu_seqlens_q, page_starts
// and kv_lens (num_seqs ints apiece) reside in host memory, page_table (table_len ints) in device
// memory. SGLang passes kv_indptr as page_starts; vLLM passes block_table (page_starts[i] = i times its
// row stride).
int cdna4_attn_paged_fwd_bf16_strided(const void* q, const void* k, const void* v, void* o, float* lse,
                                      const int64_t* strides, const int* cu_seqlens_q, const int* page_table,
                                      int table_len, const int* page_starts, const int* kv_lens, int num_seqs,
                                      int page_size, int H, int HKv, int D, int causal, float scale,
                                      hipStream_t stream);
int cdna4_attn_paged_fwd_bf16(const void* q, const void* k, const void* v, void* o, const int* cu_seqlens_q,
                              const int* page_table, int table_len, const int* page_starts,
                              const int* kv_lens, int num_seqs, int page_size, int H, int HKv, int D,
                              int causal, float scale, hipStream_t stream);
int cdna4_attn_mla_paged_fwd_bf16_strided(const void* q, const void* k, const void* v, void* o, float* lse,
                                          const int64_t* strides, const int* cu_seqlens_q,
                                          const int* page_table, int table_len, const int* page_starts,
                                          const int* kv_lens, int num_seqs, int page_size, int H, int HKv,
                                          int causal, float scale, hipStream_t stream);
int cdna4_attn_mla_paged_fwd_bf16(const void* q, const void* k, const void* v, void* o,
                                  const int* cu_seqlens_q, const int* page_table, int table_len,
                                  const int* page_starts, const int* kv_lens, int num_seqs, int page_size,
                                  int H, int HKv, int causal, float scale, hipStream_t stream);

// Varlen variant keeping cu_seqlens_q and _k in device memory, like flash_attn_varlen_func: no host
// sync and graph-capturable (a replay consumes each array's current contents). A sequence holds at most
// max_seqlen_q q rows (any excess leading rows go uncomputed) and max_seqlen_k keys; total_q and total_k
// are the tensors' token counts. host_cu_seqlens_q and _k are optional host copies of the arrays; when
// present, an eager call is planned on the host just as with the host-memory entries (a graph capture
// ignores them).
int cdna4_attn_varlen_fwd_bf16_dev_strided(const void* q, const void* k, const void* v, void* o, float* lse,
                                           const int64_t* strides, const int* cu_seqlens_q,
                                           const int* cu_seqlens_k, const int* host_cu_seqlens_q,
                                           const int* host_cu_seqlens_k, int num_seqs, int max_seqlen_q,
                                           int max_seqlen_k, int total_q, int total_k, int H, int HKv, int D,
                                           int causal, float scale, hipStream_t stream);
int cdna4_attn_mla_varlen_fwd_bf16_dev_strided(const void* q, const void* k, const void* v, void* o,
                                               float* lse, const int64_t* strides, const int* cu_seqlens_q,
                                               const int* cu_seqlens_k, const int* host_cu_seqlens_q,
                                               const int* host_cu_seqlens_k, int num_seqs, int max_seqlen_q,
                                               int max_seqlen_k, int total_q, int total_k, int H, int HKv,
                                               int causal, float scale, hipStream_t stream);
int cdna4_attn_varlen_fwd_bf16_dev(const void* q, const void* k, const void* v, void* o,
                                   const int* cu_seqlens_q, const int* cu_seqlens_k,
                                   const int* host_cu_seqlens_q, const int* host_cu_seqlens_k, int num_seqs,
                                   int max_seqlen_q, int max_seqlen_k, int total_q, int total_k, int H,
                                   int HKv, int D, int causal, float scale, hipStream_t stream);
int cdna4_attn_mla_varlen_fwd_bf16_dev(const void* q, const void* k, const void* v, void* o,
                                       const int* cu_seqlens_q, const int* cu_seqlens_k,
                                       const int* host_cu_seqlens_q, const int* host_cu_seqlens_k,
                                       int num_seqs, int max_seqlen_q, int max_seqlen_k, int total_q,
                                       int total_k, int H, int HKv, int causal, float scale,
                                       hipStream_t stream);

// Plan / run: plan a device-varlen call once per forward pass (same arguments as the _dev entries), then
// run it on every layer's tensors using the plan's heads and causal. A run follows its plan in stream
// order and must share its graph-capture state (else -1). Destroy a plan only after its runs finish and
// no graph will replay it.
void* cdna4_attn_plan_create(void);
void cdna4_attn_plan_destroy(void* plan);
int cdna4_attn_varlen_plan(void* plan, const int* cu_seqlens_q, const int* cu_seqlens_k,
                           const int* host_cu_seqlens_q, const int* host_cu_seqlens_k, int num_seqs,
                           int max_seqlen_q, int max_seqlen_k, int total_q, int total_k, int H, int HKv,
                           int D, int causal, hipStream_t stream);
int cdna4_attn_mla_varlen_plan(void* plan, const int* cu_seqlens_q, const int* cu_seqlens_k,
                               const int* host_cu_seqlens_q, const int* host_cu_seqlens_k, int num_seqs,
                               int max_seqlen_q, int max_seqlen_k, int total_q, int total_k, int H, int HKv,
                               int causal, hipStream_t stream);
int cdna4_attn_varlen_run(const void* plan, const void* q, const void* k, const void* v, void* o, float* lse,
                          const int64_t* strides, float scale, hipStream_t stream);
int cdna4_attn_mla_varlen_run(const void* plan, const void* q, const void* k, const void* v, void* o,
                              float* lse, const int64_t* strides, float scale, hipStream_t stream);

#ifdef __cplusplus
}
#endif
