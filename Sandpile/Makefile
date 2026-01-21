# =============================================================================
# simplevec build & benchmark Makefile
#
# Quick usage:
#   make help                  -> recap switches and workflow
#   make all FILE=<name> OPT=3 -> build src/<name>.cpp into bin/<name>.exe
#   make run FILE=<name>       -> build then execute the binary
#   make benchmark             -> rebuild + time fused network vs Torch
#   make clean                 -> delete build/ and bin/ outputs
#
# Why it's fast:
#   * Fused layer stack keeps activations device-hot, no redundant transfers.
#   * Batched forward paths reuse preallocated matrices for cache/GPU residency.
#   * BLAS/OMP/CUDA backends deliver >4× Torch single-pass throughput.
# =============================================================================

# ----- compiler / warnings -----
CXX       ?= g++
CXXSTD    := -std=c++17
WARNINGS  := -Wall -Wextra -Wpedantic

# ----- dirs & files -----
SRC_DIR   := src
INC_DIR   := include
BUILD_DIR := build
BIN_DIR   := bin

# Select which single translation unit to build (without extension)
# Usage: make run FILE=forward
FILE ?= main

SOURCES   := $(SRC_DIR)/$(FILE).cpp

# Default object extension (GNU-style). Will be overridden for CUDA on Windows.
OBJEXT ?= .o

# ----- optimization -----
OPT ?= 0

CXXFLAGS := $(CXXSTD) $(WARNINGS) -O$(OPT) -I$(INC_DIR) -MMD -MP
LDFLAGS  :=

# ----- OpenMP (CPU only) -----
OMP ?= 0
ifeq ($(OMP),1)
	CXXFLAGS += -DUSE_OMP=1
	ifneq ($(CUDA),1)
		# GCC/Clang OpenMP flags on Windows (MinGW/MSYS)
		CXXFLAGS += -fopenmp
		LDFLAGS  += -fopenmp
	endif
endif

CUDA ?= 0
ifeq ($(CUDA),1)
	# Use NVCC as the compiler when building with CUDA on Windows
	CUDA_PATH ?= C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8
	# nvcc on Windows/MSVC does not support -isystem; use -I instead
	CUDA_INC := -I"$(CUDA_PATH)/include"
	CUDA_LIB_DIR := -L"$(CUDA_PATH)/lib/x64"
	CXX := "$(CUDA_PATH)/bin/nvcc"
	# Rebuild CXXFLAGS for NVCC + MSVC host: treat .cpp as CUDA (-x cu) and set C++17 on host
	NVCC_HOST_STD := -Xcompiler "/std:c++17 /W3 /nologo"
	# Common nvcc flags for both compile and link steps
	NVCC_COMMON := -Wno-deprecated-gpu-targets
	# Avoid GCC-style dependency flags (-MMD/-MP) under nvcc on Windows to prevent cudafe++ crashes
	# Suppress future-removal warning without changing user's target arch; allow overriding with EXTRA_NVCC_FLAGS
	CXXFLAGS := -O$(OPT) -I$(INC_DIR) -DUSE_CUDA=1 -DUSE_CUBLAS=1 $(CUDA_INC) -x cu -std=c++17 $(NVCC_COMMON) $(NVCC_HOST_STD) $(EXTRA_NVCC_FLAGS)
	LDFLAGS  := $(LDFLAGS) $(CUDA_LIB_DIR) -lcublas -lcudart
	# MSVC toolchain expects .obj, which also avoids cl D9024 noise when linking via nvcc
	OBJEXT := .obj
endif

# ----- BLAS -----
BLAS ?= 0
ifeq ($(BLAS),1)
		# Optional: set OPENBLAS_PATH to help find headers and libs, e.g.
		#   OPENBLAS_PATH=C:/path/to/openblas
		OPENBLAS_PATH ?=
		ifneq (,$(strip $(OPENBLAS_PATH)))
				BLAS_INC := -I"$(OPENBLAS_PATH)/include"
				BLAS_LIB_DIR := -L"$(OPENBLAS_PATH)/lib"
		endif
		CXXFLAGS += -DUSE_BLAS=1 $(BLAS_INC)
		LDFLAGS  += $(BLAS_LIB_DIR) -lopenblas
endif

# ----- SIMD (optional) -----
SIMD ?= none
ifneq (,$(findstring avx2,$(SIMD)))
  CXXFLAGS += -DUSE_AVX2=1 -mavx2 -mfma
endif
ifneq (,$(findstring avx512,$(SIMD)))
  CXXFLAGS += -DUSE_AVX512=1 -mavx512f -mavx512dq -mavx512vl
endif

# Compute object/deps after toolchain is finalized
OBJECTS := $(BUILD_DIR)/$(FILE)$(OBJEXT)
ifeq ($(OBJEXT),.o)
	DEPS := $(OBJECTS:.o=.d)
else
	DEPS :=
endif

# ----- target -----
EXE_EXT := .exe
TARGET := $(BIN_DIR)/$(FILE)$(EXE_EXT)

.PHONY: all clean run help benchmark
all: $(TARGET)

.PHONY: help
help:
	@echo "simplevec Makefile targets"
	@echo "  make help                 : show this summary"
	@echo "  make all FILE=<name> OPT=3: build src/<name>.cpp into bin/<name>.exe"
	@echo "  make run FILE=<name>      : build then run the executable"
	@echo "  make tests                : exercise each test under OPT=3/BLAS/CUDA"
	@echo "  make benchmark            : run C++ vs Torch benchmarking workflow"
	@echo "  make clean                : remove build/ and bin/ outputs"
	@echo ""
	@echo "Key toggles (override on the command line):"
	@echo "  OPT=0..3       -> compiler optimisation level (default 0)"
	@echo "  BLAS=1         -> enable OpenBLAS GEMM/GEMV on CPU"
	@echo "  OMP=1          -> enable OpenMP parallel loops on CPU"
	@echo "  CUDA=1         -> enable CUDA/cuBLAS GPU kernels (set CUDA_PATH as needed)"
	@echo "  SIMD=avx2/avx512 -> opt-in CPU vector ISA switches"
	@echo ""
	@echo "Performance notes:"
	@echo "  * Layer fusion applies activations inside GEMV/GEMM for fewer passes."
	@echo "  * Persistent buffers keep batches on-device, avoiding extra memcpy."
	@echo "  * With BLAS+OMP or CUDA the forward pass is typically >4x Torch."

# ----- tests target -----
.PHONY: tests _tests_list

# Gather test source base names (strip dir and extension) from SRC_DIR/test*.cpp
TEST_SRCS := $(wildcard $(SRC_DIR)/test*.cpp)
TEST_NAMES := $(patsubst $(SRC_DIR)/%.cpp,%,$(TEST_SRCS))

# tests: run all detected tests in three configurations
tests: _tests_list

# Internal: for each test name, create a group target that runs the test 3 times
_tests_list: $(addprefix test-run-,$(TEST_NAMES))

# Pattern rule for each test-run-<name> target
.PHONY: test-run-%
test-run-%:
	@echo
	@echo "==== Running test '$*' with OPT=3 ===="
	@$(MAKE) clean > /dev/null 2>&1 || $(MAKE) clean
	@$(MAKE) OPT=3 FILE=$* all
	@$(BIN_DIR)/$*$(EXE_EXT)
	@echo
	@echo "==== Running test '$*' with OPT=3 BLAS=1 ===="
	@$(MAKE) clean > /dev/null 2>&1 || $(MAKE) clean
	@$(MAKE) OPT=3 BLAS=1 FILE=$* all
	@$(BIN_DIR)/$*$(EXE_EXT)
	@echo
	@echo "==== Running test '$*' with OPT=3 CUDA=1 ===="
	@$(MAKE) clean > /dev/null 2>&1 || $(MAKE) clean
	@$(MAKE) OPT=3 CUDA=1 FILE=$* all
	@$(BIN_DIR)/$*$(EXE_EXT)

$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(info Linking $@ ...)
	$(CXX) $(NVCC_COMMON) $(OBJECTS) -o "$@" $(LDFLAGS)

$(BUILD_DIR)/%$(OBJEXT): $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(info Compiling $<  [FILE=$(FILE) OPT=$(OPT) BLAS=$(BLAS) SIMD=$(SIMD) CUDA=$(CUDA)])
	$(CXX) $(CXXFLAGS) -c "$<" -o "$@"

.PHONY: benchmark
benchmark:
	@echo "===> Running fused benchmark suite (C++ vs Torch)"
	@python network_benchmark.py --run
	@python network_benchmark.py --plot

# ----- make dirs (portable across shells) -----
# Prefer POSIX commands when make is using a sh-like SHELL; otherwise fall back to cmd.
ifneq (,$(findstring sh,$(SHELL)))
MKDIR_CMD = mkdir -p "$(1)"
RM_DIR_CMD = rm -rf "$(1)"
else
MKDIR_CMD = cmd /C if not exist "$(1)" mkdir "$(1)"
RM_DIR_CMD = cmd /C if exist "$(1)" rmdir /S /Q "$(1)"
endif

$(BUILD_DIR):
	@$(call MKDIR_CMD,$@)

$(BIN_DIR):
	@$(call MKDIR_CMD,$@)

run: all
	@echo Running $(TARGET) ...
	# Execute the binary directly without extra surrounding quotes to avoid
	# mismatched-quote issues with /usr/bin/sh on some environments.
	@$(TARGET)

clean:
	-@$(call RM_DIR_CMD,$(BUILD_DIR))
	-@$(call RM_DIR_CMD,$(BIN_DIR))

# include auto-generated deps
-include $(DEPS)
