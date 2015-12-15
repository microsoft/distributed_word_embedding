PROJECT := $(shell readlink $(dir $(lastword $(MAKEFILE_LIST))) -f)

CXX = g++
CXXFLAGS = -O3 \
           -std=c++11 \
           -Wall \
           -Wno-sign-compare \
           -fno-omit-frame-pointer

MULTIVERSO_DIR = $(PROJECT)/multiverso
MULTIVERSO_INC = $(MULTIVERSO_DIR)/include/multiverso
MULTIVERSO_LIB = $(MULTIVERSO_DIR)/lib
THIRD_PARTY_LIB = $(MULTIVERSO_DIR)/third_party/lib

INC_FLAGS = -I$(MULTIVERSO_INC)
LD_FLAGS  = -L$(MULTIVERSO_LIB) -lmultiverso
LD_FLAGS += -L$(THIRD_PARTY_LIB) -lzmq -lmpich -lmpl
LD_FLAGS += -lpthread
  	  	
WORD_EMBEDDING_HEADERS = $(shell find $(PROJECT)/src -type f -name "*.h")
WORD_EMBEDDING_SRC     = $(shell find $(PROJECT)/src -type f -name "*.cpp")
WORD_EMBEDDING_OBJ = $(WORD_EMBEDDING_SRC:.cpp=.o)

BIN_DIR = $(PROJECT)/bin
WORD_EMBEDDING = $(BIN_DIR)/word_embedding

all: path \
	 word_embedding 

path: $(BIN_DIR)

$(BIN_DIR):
	mkdir -p $@

$(WORD_EMBEDDING): $(WORD_EMBEDDING_OBJ)
	$(CXX) $(WORD_EMBEDDING_OBJ) $(CXXFLAGS) $(INC_FLAGS) $(LD_FLAGS) -o $@

$(WORD_EMBEDDING_OBJ): %.o: %.cpp $(WORD_EMBEDDING_HEADERS) $(MULTIVERSO_INC)
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

word_embedding: path $(WORD_EMBEDDING)
	
clean:
	rm -rf $(BIN_DIR) $(WORD_EMBEDDING_OBJ)

.PHONY: all path word_embedding clean
