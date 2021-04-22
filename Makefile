EXEC := ray_tracer
SRC ?= src
BIN := bin
INCLUDE := include

# compilation
SDL_CMD = `sdl2-config --cflags --libs`
OPT ?= -O0
CC := nvcc
CFLAGS := -I ${INCLUDE} -g -c
CUCC := nvcc
CUFLAGS := -g -c -I ${INCLUDE}
LD := nvcc 
LDFLAGS := ${SDL_CMD} -g

EXT := .cc .cu .s
EXT_FILTER := ${foreach ext, ${EXT}, %${ext}}
SRC_FILES := ${filter ${EXT_FILTER}, ${wildcard ${SRC}/*}}
O_FILES := ${patsubst ${SRC}/%, ${BIN}/%.o, ${SRC_FILES}}

.PHONY=build
build: ${EXEC}

.PHONY=clean
clean:
	rm -rf ${BIN}
	rm -rf ${EXEC}

${EXEC}: ${O_FILES} 
	${LD} ${OPT} ${LDFLAGS} -o ${EXEC} ${O_FILES}

${filter %.cc.o, ${O_FILES}}: ${BIN}/%.cc.o: ${SRC}/%.cc
	mkdir -p ${dir $@}
	${CC} ${OPT} ${CFLAGS} -o $@ $<

${filter %.cu.o, ${O_FILES}}: ${BIN}/%.cu.o: ${SRC}/%.cu
	mkdir -p ${dir $@}
	${CUCC} ${OPT} ${CUFLAGS} -o $@ $<