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
	SHELL := cmd.exe
	.SHELLFLAGS := /C
	EXE_EXT := .exe
endif

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

MAIN_EXE := $(BIN_DIR)/main$(EXE_EXT)

ifeq ($(MP),1)
	CXXFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif

$(BUILD_DIR) $(BIN_DIR):
	@mkdir "$@"

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
ifeq ($(OS),Windows_NT)
	@"$(subst /,\\,$(BIN_DIR)/test_$(FILE)$(EXE_EXT))"
else
	@./$(BIN_DIR)/test_$(FILE)$(EXE_EXT)
endif
endif

compile: $(MAIN_EXE)
	@echo ">> Compilation complete. Executable: $(MAIN_EXE)"

run: $(MAIN_EXE)
	@echo ">> Running main executable..."
ifeq ($(OS),Windows_NT)
	@"$(subst /,\\,$(MAIN_EXE))"
else
	@./$(MAIN_EXE)
endif

.PHONY: test clean compile run

clean:
ifeq ($(OS),Windows_NT)
	-@cmd /C if exist "$(BUILD_DIR)" rmdir /S /Q "$(BUILD_DIR)"
	-@cmd /C if exist "$(BIN_DIR)" rmdir /S /Q "$(BIN_DIR)"
	-@cmd /C del /Q /F *.txt 2> NUL || exit 0
else
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
	@rm -f *.txt $(BUILD_DIR)/*.txt $(BIN_DIR)/*.txt
endif

-include $(OBJECTS:.o=.d)
-include $(BUILD_DIR)/test_*.d
