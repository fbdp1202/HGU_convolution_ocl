OPENCL=1

VPATH\=./src/
EXEC=HGUConvolution
OBJDIR= ./obj/

.SUFFIXES= .o .c

CC=gcc

OPTS=-Ofast
LDFLAGS= -lm -pthread
COMMON= -Iinclude/ -Isrc/
CFLAGS= -Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC

#OBJ= convolution.o layer.o
EXECOBJA = convolution.o define_cl.o image.o layer.o main.o utils.o convolution_ocl.o
EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))

ifeq ($(OPENCL), 1)
COMMON+= -DOPENCL
CFLAGS+= -DOPENCL
LDFLAGS+= -lOpenCL
endif # OPENCL

CFLAGS+=$(OPTS)

SRCS = ./src/
DEPS = $(wildcard include/*.h) Makefile

all: obj results $(SLIB) $(EXEC)

$(EXEC): $(EXECOBJ)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) 

$(OBJDIR)%.o: $(SRCS)%.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(EXEC) $(EXECOBJ)