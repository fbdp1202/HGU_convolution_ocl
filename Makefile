VPATH\=./src/
EXEC=HGUConvolution
OBJDIR=./obj/

.SUFFIXES= .o .c

CC=gcc

OPTS=-Ofast
LDFLAGS= -lm -pthread
COMMON= -Iinclude/ -Isrc/
CFLAGS= -Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC

#OBJ= convolution.o layer.o
EXECOBJA = main.o convolution.o layer.o

ifeq ($(OPENCL), 1)
COMMON+= -DOPENCL
CFLAGS+= -DOPENCL
LDFLAGS+= -lOpenCL
endif # OPENCL

CFLAGS+=$(OPTS)

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
DEPS = $(wildcard include/*.h) Makefile $(wildcard src/*.c)

all: obj results $(SLIB) $(EXEC)

$(EXEC): $(EXECOBJA)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	echo "$@ depends on EXECOBJ"

%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@
	echo "$@ depends on DEPS"

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC) $(EXECOBJ)