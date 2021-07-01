# vlasovpp compilation
PROJET = vlasovpp

include Makefile.config

DEBUG_MODE ?= N
PARALLEL_MODE ?= N
JC_condition  ?= 1
Bxy_condition ?= 1
Exy_condition ?= 1

SRC_dir    = source
HEADER_dir = include/$(PROJET)
OBJ_dir    = obj

# variables of Makefile.config if it isn't already define
echo    ?= echo
CXX     ?= cc
LIB_dir ?=  $(shell pwd)/lib

INC_BOOST ?= /usr/include/boost
INC_FFTW  ?= /usr/include
LIB_FFTW  ?= fftw3

INCLUDES = $(INC_BOOST) $(INC_FFTW) $(HEADER_dir)/..
LIBS = $(PROJET) $(LIB_FFTW) m

### Lists of files and flags for compilation ##########################
SRC = $(wildcard $(SRC_dir)/*.cc)
OBJ = $(SRC:$(SRC_dir)/%.cc=$(OBJ_dir)/%.o)
INC = $(wildcard $(HEADER_dir)/*.h)

GLLIBS = -lm

# check GCC version for flags
CXX_v = $(shell $(CXX) -dumpversion | awk -F '.' '{ print $$1 }')
CXX_vleq8 = $(shell [[ $(CXX_v) -le 8 ]] && echo "True" || echo "False")
ifeq ($(CXX_vleq8),True)
    GLLIBS += -lstdc++fs
endif

ifeq ($(DEBUG_MODE),Y)
	CFLAGS += -O0 -DDEBUG_MODE -Wall
	DEBUGFLAGS = -p
	
	COMPIL_mode = "\033[1mCompilation en mode debug\033[0m"
else
	CFLAGS += -O3 -DBOOST_DISABLE_ASSERTS -DNDEBUG

	COMPIL_mode = "\033[1mCompilation en mode release\033[0m"
endif

ifeq ($(PARALLEL_MODE),Y)
	LIB_FFTW = fftw3_omp fftw3
	CFLAGS += -fopenmp
	DEBUGFLAGS += -D_GLIBCXX_PARALLEL
endif

#linker
LD = $(CXX) -fPIC -shared
LDFLAGS += $(CFLAGS)

### BUILD RULES #######################################################
all : $(LIB_dir)/lib$(PROJET).so
	@$(echo) "all your base are belong to us"
	@$(echo) "Library build : \033[90mlib$(PROJET).so\033[0m"

$(LIB_dir)/lib$(PROJET).so : $(LIB_dir) $(OBJ_dir) $(OBJ)
	@$(echo) $(COMPIL_mode)
	@$(echo) "\033[36m$(notdir $@)\033[0m"
	$(LD) $(OBJ) $(LDFLAGS) $(GLLIBS) -o $@
	@$(echo) "\033[90mCompilation finie.\033[0m"

$(OBJ_dir)/%.o : $(SRC_dir)/%.cc $(HEADER_dir)/%.h
	@$(echo) "\033[95m$@\033[0m"
	$(CXX) $(CFLAGS) -fPIC $(foreach inc,$(INCLUDES),-I $(inc)) -o $@ -c $<

$(OBJ_dir) :
	@$(echo) "\033[92mmake $@ directory\033[0m"
	mkdir -p $@
$(LIB_dir) :
	@$(echo) "\033[92mmake $@ directory\033[0m"
	mkdir -p $@

## SCAN-BUILD #########################################################
EXE = nuke hybrid1dx3dv_lawson_filtre.out
scan-build : nuke
	scan-build -o html \
		-enable-checker debug.ViewCallGraph \
		--use-c++=/usr/local/Cellar/llvm/11.0.0/bin/clang++ \
	  --use-analyzer=/usr/local/Cellar/llvm/11.0.0/bin/clang \
		-stats \
		make $(EXE)


### cleanning rules

clean :
	@$(echo) "\033[41;97m ** Suppression des fichiers objets ** \033[0m"
	rm -f $(OBJ_dir)/*.o

mrproper : clean
	@$(echo) "\033[31;1m Suppression de la bibliothèque \033[0m"
	rm -f $(LIB_dir)/lib$(PROJET).so

nuke : mrproper
	@$(echo) "  ,-*\n (_)\n"
	rm -f *.out
	rm -Rf *.out.dSYM

coffee :
	@$(echo) "It's coffee time!\n"
	@$(echo) " (\n  )\nc[]"

### compilation rules

%.out : %.cc $(LIB_dir)/lib$(PROJET).so $(INC)
	@$(echo) "\033[95m$@\033[0m"
	$(CXX) $(CFLAGS) $(DEBUGFLAGS) -g $< $(foreach inc,$(INCLUDES),-I $(inc)) -L $(LIB_dir) $(foreach lib,$(LIBS),-l$(lib)) -D JC_condition=$(JC_condition) -D Bxy_condition=$(Bxy_condition) -D Exy_condition=$(Exy_condition) -o $@
	@$(echo) "\033[1mcompilation done\033[0m"
