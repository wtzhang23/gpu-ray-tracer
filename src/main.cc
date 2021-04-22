#include <iostream>
#include <SDL2/SDL.h>
#include <memory>
#include <vector>
#include <thread>
#include <string>
#include <cstdio>
#include <atomic>
#include "raytracer.h"
#include "linear.h"
#include "canvas.h"
#include "scene.h"

const int WIDTH = 640;
const int HEIGHT = 480;

int main(int argc, const char** argv) {
    SDL_Init(SDL_INIT_VIDEO);
    std::shared_ptr<std::atomic_bool> running = std::shared_ptr<std::atomic_bool>(new std::atomic_bool(true));
    
    canvas::Color col = canvas::Color(1.0, .5, 1.0);
    std::cout << col << std::endl;
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
    canvas::Canvas canvas{WIDTH, HEIGHT};
    scene::Scene scene{canvas};
    int rmask = canvas::Color::rmask();
    int gmask = canvas::Color::gmask();
    int bmask = canvas::Color::bmask();
    int amask = canvas::Color::amask();
    SDL_Surface* surface = NULL;
    canvas.get_surface(&surface);
    assert(surface != NULL);
    while (running->load()) {
        raytracer::update_scene(scene);
        int rv = SDL_BlitSurface(surface, NULL, screen, NULL);
        assert(rv == 0);
        rv = SDL_UpdateWindowSurface(window);
        assert(rv == 0);
    }
    SDL_FreeSurface(surface);

    io_thread.join();
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}