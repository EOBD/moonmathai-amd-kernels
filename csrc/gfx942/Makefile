
ROCM_PATH  ?= /opt/rocm
HIPCC      ?=  $(ROCM_PATH)/bin/hipcc
# HIPCC      ?=  /mnt/win/Users/evyat/Documents/Dev/inferevya/inferevya/Tools/hipcc_nolicm.py
# HIPCC      ?= evyampiler
GPU_ARCH   ?= gfx942
MLA_HIPCC  ?= $(HIPCC)

# evyampiler: two compounding fixes found with Tools/spill_bisect.py —
#  1. mid-end LICM hoists the persistent loop's threadIdx-derived address set
#     (same failure as the MLP kernel): use the nolicm pipeline.
#  2. AMDGPUAttributor's regalloc hints ("amdgpu-agpr-alloc", "amdgpu-no-*")
#     take the isLite variants from 110/129 -> 155/151 spill instructions at
#     the very end of the mid-end (pass N=34431). They HELP the dense variants,
#     so the scrub is targeted at the lite kernels only.
# Net: 396 -> 287 scratch instructions across the 6 attention_forward variants.
# EVYAMPILER_PIPELINE_FILE ?= /mnt/win/Users/evyat/Documents/Dev/inferevya/inferevya/evyampiler/pipeline_nolicm.txt
# EVYAMPILER_SCRUB_ATTRS   ?= attention_forwardILb0ELb1ELb0E,attention_forwardILb1ELb1ELb0E
# export EVYAMPILER_PIPELINE_FILE EVYAMPILER_SCRUB_ATTRS

CXXFLAGS_BASE   = -O3 -ffast-math --offload-arch=$(GPU_ARCH) -std=c++20 -Wno-unused-result
INCLUDES        = -I. -I$(ROCM_PATH)/include -I$(ROCM_PATH)/include/hip

# -ffast-math is REQUIRED

# DWARF symbols + ATT source-mapping — KERNEL OBJECTS ONLY.
#   $(CURDIR)=. rewrites a kernel's own path to a relative path so
#   rocprofv3's --att snapshotter resolves it relative to this dir (which main.sh
#   anchors via `cd $SCRIPT_DIR`) and copies it into snapshots.json.
#   opus.hpp is redirected to a non-existent sentinel so it is deliberately
#   EXCLUDED from the opcode→source view. clang applies -fdebug-prefix-map rules
#   in order and the LAST matching prefix wins, so the opus.hpp map MUST come
#   AFTER the .=. map.
KERNEL_DWARF_FLAGS = -g3 -gdwarf-5 -fdebug-info-for-profiling -gno-pubnames \
                     -fdebug-prefix-map=$(CURDIR)=. \
                     -fdebug-prefix-map=$(CURDIR)/opus.hpp=/excluded-from-att-source-view/opus.hpp

# release tester, no debug symbols (keeps the device code object lean and
# ensures DWARF is kernel-only).

BUILD_DIR   = build
KERNEL_OBJ  = $(BUILD_DIR)/attention_kernel.o
TESTER_OBJ  = $(BUILD_DIR)/attention_tester.o
TARGET      = $(BUILD_DIR)/attention_tester

KERNEL_SRC  = attention_kernel.hip
TESTER_SRC  = attention_tester.cpp
KERNEL_HDR  = attention_kernel.h
OPUS_HDR    = opus.hpp

MLA_KERNEL_OBJ = $(BUILD_DIR)/mla_decode_a16w8.o
MLA_TESTER_OBJ = $(BUILD_DIR)/mla_decode_a16w8_tester.o
MLA_TARGET     = $(BUILD_DIR)/mla_decode_a16w8_tester

MLA_KERNEL_SRC = mla_decode_a16w8.hip
MLA_TESTER_SRC = mla_decode_a16w8_tester.cpp

# -fno-math-errno keeps sqrtf/expf a compiler builtin (no glibc call), so the
# binary stays portable across glibc versions (see ~/.claude memory: build
# portability; MLA needs it explicitly since it lacks -ffast-math's implicit
# errno suppression).
MLA_CXXFLAGS = -O3 -ffast-math -fno-math-errno --offload-arch=$(GPU_ARCH) -std=c++20 -Wno-unused-result

# ── Dependency tracking ───────────────────────────────────────────────────────
# Track this Makefile itself: any flag/rule change triggers a full rebuild.
MAKEFILE_DEPS := $(MAKEFILE_LIST)

# All compilation units transitively depend on every header (kernel.h includes opus.hpp).
ALL_HDRS = $(KERNEL_HDR) $(OPUS_HDR)

.DELETE_ON_ERROR:

.PHONY: all attention mla clean rebuild

all: $(TARGET) $(MLA_TARGET)

# Build just the dense attention tester.
attention: $(TARGET)

# Build just the MLA A16W8 tester (used by main_mla.sh / `make mla`).
mla: $(MLA_TARGET)

# Force-rebuild (ignores timestamps, always recompiles).
rebuild: clean all

$(TARGET): $(KERNEL_OBJ) $(TESTER_OBJ) $(MAKEFILE_DEPS) | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS_BASE) -o $@ $(KERNEL_OBJ) $(TESTER_OBJ)

$(KERNEL_OBJ): $(KERNEL_SRC) $(ALL_HDRS) $(MAKEFILE_DEPS) | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS_BASE) $(KERNEL_DWARF_FLAGS) $(INCLUDES) -c -o $@ $<

$(TESTER_OBJ): $(TESTER_SRC) $(ALL_HDRS) $(MAKEFILE_DEPS) | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS_BASE) $(INCLUDES) -c -o $@ $<

$(MLA_TARGET): $(MLA_KERNEL_OBJ) $(MLA_TESTER_OBJ) $(MAKEFILE_DEPS) | $(BUILD_DIR)
	$(MLA_HIPCC) $(MLA_CXXFLAGS) -o $@ $(MLA_KERNEL_OBJ) $(MLA_TESTER_OBJ)

$(MLA_KERNEL_OBJ): $(MLA_KERNEL_SRC) $(MAKEFILE_DEPS) | $(BUILD_DIR)
	$(MLA_HIPCC) $(MLA_CXXFLAGS) $(KERNEL_DWARF_FLAGS) $(INCLUDES) -c -o $@ $<

$(MLA_TESTER_OBJ): $(MLA_TESTER_SRC) $(MAKEFILE_DEPS) | $(BUILD_DIR)
	$(MLA_HIPCC) $(MLA_CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
