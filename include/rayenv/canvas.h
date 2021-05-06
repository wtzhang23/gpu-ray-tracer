#ifndef RAYENV_CANVAS_H
#define RAYENV_CANVAS_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <cstdint>
#include <SDL2/SDL.h>
#include <iostream>
#include "rayenv/color.h"
#include "raymath/linear.h"

namespace renv {
class Canvas {
private:
    int* buffer;
    int width;
    int height;
public:
    Canvas(int width, int height);

    CUDA_HOSTDEV
    void set_color(int x, int y, const Color& color);

    CUDA_HOSTDEV
    Color get_color(int x, int y);

    friend void free_canvas(Canvas& canvas);

    void get_surface(SDL_Surface** surface);

    CUDA_HOSTDEV
    int get_height() const {
        return height;
    }

    CUDA_HOSTDEV
    int get_width() const {
        return width;
    }
};
}

#endif