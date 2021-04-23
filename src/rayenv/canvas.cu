#include "rayenv/canvas.h"
#include "rayenv/color.h"
#include "gputils/alloc.h"


namespace renv {
Canvas::Canvas(int width, int height): buffer((int*) gputils::create_buffer(width * height, sizeof(int))), width(width), height(height){}

CUDA_HOSTDEV
void Canvas::set_color(int x, int y, const Color& color) {
    buffer[y * width + x] = color.to_encoding();
}

CUDA_HOSTDEV
Color Canvas::get_color(int x, int y) {
    return Color::from_encoding(buffer[y * width + x]);
}

void free_canvas(Canvas& canvas) {
    gputils::free_buffer((void*) canvas.buffer);
}

void Canvas::get_surface(SDL_Surface** surface) {
    *surface = SDL_CreateRGBSurfaceFrom(buffer, width, height, sizeof(int) * 8, sizeof(int) * width, 
                Color::rmask(),
                Color::gmask(),
                Color::bmask(),
                Color::amask());
}
}