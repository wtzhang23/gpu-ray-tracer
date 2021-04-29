#include <iostream>
#include <SDL2/SDL.h>
#include <memory>
#include <vector>
#include <thread>
#include <string>
#include <cstdio>
#include <atomic>
#include <cmath>
#include <cuda.h>
#include <png.h>
#include "assets.h"
#include "raytracer.h"
#include "raymath/linear.h"
#include "rayenv/canvas.h"
#include "rayenv/scene.h"
#include "raymath/geometry.h"
#include "rayprimitives/entity.h"
#include "rayprimitives/trimesh.h"

static const int WIDTH = 640;
static const int HEIGHT = 480;
static const char* ATLAS_PATH = "assets/sus.png";

int main(int argc, const char** argv) {
    renv::Canvas canvas{WIDTH, HEIGHT};
    renv::Camera camera{M_PI / 4, canvas};

    // load assets
    gputils::TextureBuffer4D<float> atlas = assets::read_png(ATLAS_PATH);

    // build hitables
    std::vector<rmath::Vec3<float>> vertices;
    std::vector<rmath::Vec3<float>> normals;

    std::vector<rprimitives::Hitable*> hitables;
    renv::Scene scene{canvas, camera, atlas, hitables};
    int rmask = renv::Color::rmask();
    int gmask = renv::Color::gmask();
    int bmask = renv::Color::bmask();
    int amask = renv::Color::amask();
    
    // initialize window
    SDL_Init(SDL_INIT_VIDEO);
    std::shared_ptr<std::atomic_bool> running = std::shared_ptr<std::atomic_bool>(new std::atomic_bool(true));
    
    auto io_thread = std::thread([=]{
        while (running->load()) {
            SDL_Event event;
            if (SDL_WaitEvent(&event)){
                switch (event.type) {
                    case SDL_QUIT:
                        running->store(false);
                        break;
                }
            }
        }
    });

    SDL_Window* window = SDL_CreateWindow("Ray Tracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 
                                                            WIDTH, HEIGHT, SDL_WINDOW_RESIZABLE);
    SDL_Surface* screen = SDL_GetWindowSurface(window);
    SDL_Surface* surface = NULL;
    canvas.get_surface(&surface);
    assert(surface != NULL);
    while (running->load()) {
        rtracer::update_scene(scene);
        int rv = SDL_BlitSurface(surface, NULL, screen, NULL);
        assert(rv == 0);
        rv = SDL_UpdateWindowSurface(window);
        assert(rv == 0);
    }
    io_thread.join();
    SDL_FreeSurface(surface);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}