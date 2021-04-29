EXEC := ray_tracer
SRC ?= src
INCLUDE := include

# compilation
SDL_CMD = `sdl2-config --cflags --libs`
OPT ?= -O0
CUCC := nvcc
CUFLAGS := -g -G -rdc=true ${SDL_CMD} -lpng

EXT := .cc .cu .s
EXT_FILTER := ${foreach ext, ${EXT}, %${ext}}
SRC_FILES := ${filter ${EXT_FILTER}, ${shell find ${SRC}}}

.PHONY=build
build: ${EXEC}

.PHONY=clean
clean:
	rm -rf ${EXEC}

${EXEC}: ${SRC_FILES} 
	${CUCC} -I ${INCLUDE} -o $@ ${OPT} ${CUFLAGS} ${O_FILES} ${SRC_FILES}
