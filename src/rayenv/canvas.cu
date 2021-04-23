#include "rayenv/canvas.h"
#include "gpu_alloc.h"


namespace renv {
CUDA_HOSTDEV
unsigned int Color::rshift() {
    return R_ORDER << 3;
}

CUDA_HOSTDEV
unsigned int Color::gshift() {
    return G_ORDER << 3;
}

CUDA_HOSTDEV
unsigned int Color::bshift() {
    return B_ORDER << 3;
}

CUDA_HOSTDEV
unsigned int Color::ashift() {
    return A_ORDER << 3;
}

CUDA_HOSTDEV
int Color::to_encoding() const {
    return (((int) inner[0]) << rshift()) + (((int) inner[1]) << gshift()) + (((int) inner[2]) << bshift()) + (((int) inner[3]) << ashift());
}

CUDA_HOSTDEV
Color Color::from_encoding(int encoding) {
    std::uint8_t r = (std::uint8_t) (((encoding & Color::rmask()) >> Color::rshift()) & UINT8_MAX);
    std::uint8_t g = (std::uint8_t) (((encoding & Color::gmask()) >> Color::gshift()) & UINT8_MAX);
    std::uint8_t b = (std::uint8_t) (((encoding & Color::bmask()) >> Color::bshift()) & UINT8_MAX);
    std::uint8_t a = (std::uint8_t) (((encoding & Color::amask()) >> Color::ashift()) & UINT8_MAX);
    return Color(r, g, b, a);
}

CUDA_HOSTDEV
int Color::rmask() {
    return ((int) UINT8_MAX) << Color::rshift();
}

CUDA_HOSTDEV
int Color::gmask() {
    return ((int) UINT8_MAX) << Color::gshift();
}

CUDA_HOSTDEV
int Color::bmask() {
    return ((int) UINT8_MAX) << Color::bshift();
}

CUDA_HOSTDEV
int Color::amask() {
    return ((int) UINT8_MAX) << Color::ashift();
}

CUDA_HOSTDEV
std::uint8_t Color::r() const {
    return inner[0];
}

CUDA_HOSTDEV
std::uint8_t Color::g() const {
    return inner[1];
}

CUDA_HOSTDEV
std::uint8_t Color::b() const {
    return inner[2];
}

CUDA_HOSTDEV
std::uint8_t Color::a() const {
    return inner[3];
}

std::ostream& operator<<(std::ostream& os, const Color& col) {
    os << "[r: " << (int) col.r() << ", g: " << (int) col.g() << ", b: " << (int) col.b() << ", a: " << (int) col.a() << "]"; 
    return os;
}

Canvas::Canvas(int width, int height): buffer((int*) gpu_alloc::create_buffer(width * height, sizeof(int))), width(width), height(height){}

CUDA_HOSTDEV
void Canvas::set_color(int x, int y, const Color& color) {
    buffer[y * width + x] = color.to_encoding();
}

CUDA_HOSTDEV
Color Canvas::get_color(int x, int y) {
    return Color::from_encoding(buffer[y * width + x]);
}

void free_canvas(Canvas& canvas) {
    gpu_alloc::free_buffer((void*) canvas.buffer);
}

void Canvas::get_surface(SDL_Surface** surface) {
    *surface = SDL_CreateRGBSurfaceFrom(buffer, width, height, sizeof(int) * 8, sizeof(int) * width, 
                Color::rmask(),
                Color::gmask(),
                Color::bmask(),
                Color::amask());
}
}