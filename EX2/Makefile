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

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

MAIN_EXE := $(BIN_DIR)/main$(EXE_EXT)

$(BUILD_DIR) $(BIN_DIR):
	@mkdir -p $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(MAIN_EXE): $(BUILD_DIR)/main.o $(OBJECTS) | $(BIN_DIR)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_%.o: $(TEST_DIR)/test_%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/test_%$(EXE_EXT): $(BUILD_DIR)/test_%.o $(OBJECTS) | $(BIN_DIR)
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

compile: $(MAIN_EXE)
	@echo ">> Compilation complete. Executable: $(MAIN_EXE)"

run: $(MAIN_EXE)
	@echo ">> Running main executable..."
	@./$(MAIN_EXE)

.PHONY: test clean compile run

clean:
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
	@rm -f *.txt $(BUILD_DIR)/*.txt $(BIN_DIR)/*.txt

-include $(OBJECTS:.o=.d)
-include $(BUILD_DIR)/test_*.d
