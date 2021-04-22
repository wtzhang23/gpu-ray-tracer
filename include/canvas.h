#ifndef CANVAS_H
#define CANVAS_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <cstdint>
#include <SDL2/SDL.h>
#include <iostream>
#include "linear.h"

namespace canvas {
class Color {
private:
    linear::Vec4<std::uint8_t> inner;
    const static int R_ORDER = 3;
    const static int G_ORDER = 2;
    const static int B_ORDER = 1;
    const static int A_ORDER = 0;

    CUDA_HOSTDEV
    static unsigned int rshift() {
        return R_ORDER << 3;
    }

    CUDA_HOSTDEV
    static unsigned int gshift() {
        return G_ORDER << 3;
    }

    CUDA_HOSTDEV
    static unsigned int bshift() {
        return B_ORDER << 3;
    }

    CUDA_HOSTDEV
    static unsigned int ashift() {
        return A_ORDER << 3;
    }
public:
    CUDA_HOSTDEV
    Color(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a): inner({r, g, b, a}) {}
    CUDA_HOSTDEV
    Color(std::uint8_t r, std::uint8_t g, std::uint8_t b): Color(r, g, b, UINT8_MAX) {}
    CUDA_HOSTDEV
    Color(double r, double g, double b, double a): Color((std::uint8_t) ((double) UINT8_MAX * r), (std::uint8_t) ((double) UINT8_MAX * g), 
                                                    (std::uint8_t) ((double) UINT8_MAX * b), (std::uint8_t) ((double) UINT8_MAX * a)) {}
    CUDA_HOSTDEV
    Color(float r, float g, float b, float a): Color((std::uint8_t) ((float) UINT8_MAX * r), (std::uint8_t) ((float) UINT8_MAX * g), 
                                                    (std::uint8_t) ((float) UINT8_MAX * b), (std::uint8_t) ((float) UINT8_MAX * a)) {}
    CUDA_HOSTDEV
    Color(double r, double g, double b): Color(r, g, b, 1.0){}
    CUDA_HOSTDEV
    Color(float r, float g, float b): Color(r, g, b, 1.0f){}

    CUDA_HOSTDEV
    int to_encoding() const {
        return (((int) inner[0]) << rshift()) + (((int) inner[1]) << gshift()) + (((int) inner[2]) << bshift()) + (((int) inner[3]) << ashift());
    }

    CUDA_HOSTDEV
    static Color from_encoding(int encoding) {
        std::uint8_t r = (std::uint8_t) (((encoding & Color::rmask()) >> Color::rshift()) & UINT8_MAX);
        std::uint8_t g = (std::uint8_t) (((encoding & Color::gmask()) >> Color::gshift()) & UINT8_MAX);
        std::uint8_t b = (std::uint8_t) (((encoding & Color::bmask()) >> Color::bshift()) & UINT8_MAX);
        std::uint8_t a = (std::uint8_t) (((encoding & Color::amask()) >> Color::ashift()) & UINT8_MAX);
        return Color(r, g, b, a);
    }

    CUDA_HOSTDEV
    static int rmask() {
        return ((int) UINT8_MAX) << Color::rshift();
    }

    CUDA_HOSTDEV
    static int gmask() {
        return ((int) UINT8_MAX) << Color::gshift();
    }

    CUDA_HOSTDEV
    static int bmask() {
        return ((int) UINT8_MAX) << Color::bshift();
    }

    CUDA_HOSTDEV
    static int amask() {
        return ((int) UINT8_MAX) << Color::ashift();
    }

    CUDA_HOSTDEV
    std::uint8_t r() const {
        return inner[0];
    }

    CUDA_HOSTDEV
    std::uint8_t g() const {
        return inner[1];
    }

    CUDA_HOSTDEV
    std::uint8_t b() const {
        return inner[2];
    }

    CUDA_HOSTDEV
    std::uint8_t a() const {
        return inner[3];
    }

    friend std::ostream& operator<<(std::ostream& os, const Color& col) {
        os << "[r: " << (int) col.r() << ", g: " << (int) col.g() << ", b: " << (int) col.b() << ", a: " << (int) col.a() << "]"; 
        return os;
    }
};

int* create_buffer(int width, int height);
void free_buffer(int* buffer);

class Canvas {
private:
    int* buffer;
    int width;
    int height;
    Canvas(int* buffer, int width, int height): buffer(buffer), width(width), height(height){}
public:
    Canvas(int width, int height): Canvas(create_buffer(width, height), width, height){}
    Canvas& operator=(const Canvas& canvas) {
        buffer = canvas.buffer;
        width = canvas.width;
        height = canvas.height;
        return *this;
    };

    CUDA_HOSTDEV
    ~Canvas() {}

    CUDA_HOSTDEV
    void set_color(int row, int col, const Color& color) {
        buffer[row * width + col] = color.to_encoding();
    }

    CUDA_HOSTDEV
    Color get_color(int row, int col) {
        return Color::from_encoding(buffer[row * width + col]);
    }

    friend void free_canvas(Canvas& canvas) {
        free_buffer(canvas.buffer);
    }

    void get_surface(SDL_Surface** surface) {
        *surface = SDL_CreateRGBSurfaceFrom(buffer, width, height, sizeof(int) * 8, sizeof(int) * width, 
                    Color::rmask(),
                    Color::gmask(),
                    Color::bmask(),
                    Color::amask());
    }

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