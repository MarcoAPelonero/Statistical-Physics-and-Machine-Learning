CXX       ?= g++
CXXFLAGS  ?= -std=c++17 -O2 -Wall -Wextra -Iinclude -MMD -MP
LDFLAGS   ?=
SRC_DIR    = src
INC_DIR    = include
TEST_DIR   = tests
BUILD_DIR  = build
BIN_DIR    = bin

EXE_EXT :=
ifeq ($(OS),Windows_NT)
  EXE_EXT := .exe
endif

$(BUILD_DIR) $(BIN_DIR):
	@mkdir -p $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_%.o: $(TEST_DIR)/test_%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/test_%$(EXE_EXT): $(BUILD_DIR)/%.o $(BUILD_DIR)/test_%.o | $(BIN_DIR)
	$(CXX) $^ -o $@ $(LDFLAGS)

ifndef FILE
test:
	@echo "Usage: make test FILE=<name>"; \
	echo "Example: make test FILE=rng"; \
	exit 2
else
test: $(BIN_DIR)/test_$(FILE)$(EXE_EXT)
	@echo ">> Running test for $(FILE)..."
	@./$(BIN_DIR)/test_$(FILE)$(EXE_EXT)
endif

.PHONY: test clean

clean:
	@rm -rf $(BUILD_DIR) $(BIN_DIR)