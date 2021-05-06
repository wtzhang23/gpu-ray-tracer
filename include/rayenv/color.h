#ifndef RAYENV_COLOR_H
#define RAYENV_COLOR_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <cstdint>
#include <SDL2/SDL.h>
#include <iostream>
#include "raymath/linear.h"

namespace renv {
class Color {
private:
    rmath::Vec4<std::uint8_t> inner;
    const static int R_ORDER = 3;
    const static int G_ORDER = 2;
    const static int B_ORDER = 1;
    const static int A_ORDER = 0;

    CUDA_HOSTDEV
    static unsigned int rshift();
    CUDA_HOSTDEV
    static unsigned int gshift();
    CUDA_HOSTDEV
    static unsigned int bshift();
    CUDA_HOSTDEV
    static unsigned int ashift();
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
    template<typename T>
    CUDA_HOSTDEV
    Color(rmath::Vec3<T> v): Color(v[0], v[1], v[2]){}

    CUDA_HOSTDEV
    int to_encoding() const;

    CUDA_HOSTDEV
    static Color from_encoding(int encoding);

    CUDA_HOSTDEV
    static int rmask();

    CUDA_HOSTDEV
    static int gmask();

    CUDA_HOSTDEV
    static int bmask();

    CUDA_HOSTDEV
    static int amask();

    CUDA_HOSTDEV
    std::uint8_t r() const;

    CUDA_HOSTDEV
    std::uint8_t g() const;

    CUDA_HOSTDEV
    std::uint8_t b() const;

    CUDA_HOSTDEV
    std::uint8_t a() const;

    friend std::ostream& operator<<(std::ostream& os, const Color& col);
};
}

#endif