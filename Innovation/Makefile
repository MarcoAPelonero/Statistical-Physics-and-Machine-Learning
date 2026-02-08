.PHONY: all clean main tests run-tests run

# Compiler and flags
CXX := g++
CXXFLAGS_BASE := -std=c++17 -Wall -Wextra

# Production flags for main.cpp (O0: MinGW has known issues with -O1/-O2/-O3)
CXXFLAGS_PROD := $(CXXFLAGS_BASE) -g -O0 -fopenmp

# Test flags — O0 for debugging (MinGW has known issues with -O1/-O2/-O3)
CXXFLAGS_TEST := $(CXXFLAGS_BASE) -g -O0 -fopenmp

# Include directories
INCLUDE := -Iinclude

# Output directories
BUILD_DIR := build
BIN_DIR := bin

# Source files
MAIN_SRC := main.cpp
TEST_SRC := $(wildcard tests/*.cpp)
TEST_EXES := $(TEST_SRC:tests/%.cpp=$(BIN_DIR)/test_%)

# Default target
all: main

# Create directories
$(BUILD_DIR) $(BIN_DIR):
	@mkdir -p $@

# Build main executable with O3 optimization
main: $(BIN_DIR) $(BIN_DIR)/main

$(BIN_DIR)/main: $(MAIN_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS_PROD) $(INCLUDE) -o $@ $<

# Build test executables
tests: $(TEST_EXES)

$(BIN_DIR)/test_%: tests/%.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS_TEST) $(INCLUDE) -o $@ $<

# Run all tests
run-tests: tests
	@for test in $(TEST_EXES); do \
		echo "Running $$test..."; \
		$$test || exit 1; \
	done

# Run the full pipeline
run-pipeline: main
	@$(BIN_DIR)/main

# Run a single test file (usage: make run FILE=test_name)
# Example: make run FILE=test_name (runs bin/test_test_name)
run: $(BIN_DIR)/test_$(FILE)
	@$(BIN_DIR)/test_$(FILE)

# Clean build artifacts
clean:
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Cleaned build artifacts"
