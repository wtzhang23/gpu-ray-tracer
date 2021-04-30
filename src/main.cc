#include <iostream>
#include <SDL2/SDL.h>
#include <thread>
#include <atomic>
#include "raytracer.h"
#include "rayenv/canvas.h"

static const int WIDTH = 640;
static const int HEIGHT = 480;

int main(int argc, const char** argv) {
    renv::Scene scene = rtracer::build_scene(WIDTH, HEIGHT);
    
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
    scene.get_canvas().get_surface(&surface); // link scene near plane to screen
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