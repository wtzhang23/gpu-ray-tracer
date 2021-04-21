EXEC := ray_tracer
SRC ?= src
BIN := bin
INCLUDE := include
CC := nvcc
CFLAGS := -I ${INCLUDE} -O3 -g -c
CUCC := nvcc
CUFLAGS := -g -c -I ${INCLUDE} -O3
LD := nvcc
LDFLAGS := -O3

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
	${LD} ${LDFLAGS} -o ${EXEC} ${O_FILES}

${filter %.cc.o, ${O_FILES}}: ${BIN}/%.cc.o: ${SRC}/%.cc
	mkdir -p ${dir $@}
	${CC} ${CFLAGS} -o $@ $<

${filter %.cu.o, ${O_FILES}}: ${BIN}/%.cu.o: ${SRC}/%.cu
	mkdir -p ${dir $@}
	${CUCC} ${CUFLAGS} -o $@ $<