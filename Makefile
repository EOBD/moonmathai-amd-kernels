# Direct kernel builds, without torch.
#
# CDNA3 (gfx942) attention forward:
#           make                                   — build both RTNE and RTZ variants
#           make rtne                              — build the RTNE pack only
#           make rtz                               — build the RTZ pack only
#           make EXTRA_LLVM_FLAGS="--save-temps"   — keep the IR/asm intermediates
#
# CDNA4 (gfx950) attention forward, C API in csrc/gfx950/attention.h:
#           make gfx950                            — build libattention_gfx950.so
#
# Run:      python runner.py                       — benchmark HIP RTNE/RTZ against AITER

HIPCC ?= hipcc
ARCH = gfx942
GFX950_ARCH ?= gfx950

TARGET_RTNE = libattention_rtne.so
TARGET_RTZ  = libattention_rtz.so
SRC         = csrc/gfx942/attention_kernel.hip

TARGET_GFX950 = libattention_gfx950.so
SRC_GFX950    = csrc/gfx950/attention.hip csrc/gfx950/attention_mla.hip
DEPS_GFX950   = $(wildcard csrc/gfx950/*.h csrc/gfx950/*.hpp csrc/gfx950/opus/*.hpp)

EXTRA_LLVM_FLAGS ?=
HIPCC_FLAGS = -shared -fPIC -O3 --offload-arch=$(ARCH) \
              -ffast-math -fno-math-errno \
              -mllvm -amdgpu-early-inline-all=true \
              -Icsrc/gfx942 \
              $(EXTRA_LLVM_FLAGS)

# -fno-slp-vectorize: SLP would pack the softmax into v_pk_*_f32, which can't co-issue with MFMA.
HIPCC_FLAGS_GFX950 = -shared -fPIC -O3 -std=c++17 --offload-arch=$(GFX950_ARCH) \
                     -ffast-math -fno-math-errno -fno-slp-vectorize \
                     -mllvm -amdgpu-early-inline-all=true \
                     -Icsrc/gfx950 \
                     $(EXTRA_LLVM_FLAGS)

all: $(TARGET_RTNE) $(TARGET_RTZ)

rtne: $(TARGET_RTNE)
rtz:  $(TARGET_RTZ)
gfx950: $(TARGET_GFX950)

$(TARGET_RTNE): $(SRC)
	$(HIPCC) $(HIPCC_FLAGS) -DBF16_ROUND=RTNE -o $@ $<

$(TARGET_RTZ): $(SRC)
	$(HIPCC) $(HIPCC_FLAGS) -DBF16_ROUND=RTZ -o $@ $<

$(TARGET_GFX950): $(SRC_GFX950) $(DEPS_GFX950)
	$(HIPCC) $(HIPCC_FLAGS_GFX950) -o $@ $(SRC_GFX950)

clean:
	rm -f libattention*.so \
	      attention_kernel*-hip-amdgcn-amd-amdhsa-gfx942.* \
	      attention_kernel*-host-x86_64-unknown-linux-gnu.* \
	      attention_kernel*.hip-hip-amdgcn-amd-amdhsa.hipfb \
	      attention*-hip-amdgcn-amd-amdhsa-gfx950.* \
	      attention*-host-x86_64-unknown-linux-gnu.*

.PHONY: all rtne rtz gfx950 clean
