#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <chrono>
#include <sstream>
#include <iostream>
#include <cuda.h>
#include "raytracer.h"
#include "rayenv/canvas.h"
#include "raymath/geometry.h"
#include "raymath/linear.h"
#include "rayenv/scene.h"
#include "procedural/cube_world.h"

static const int WIDTH = 640;
static const int HEIGHT = 480;
static const float MOVE_SPEED = 0.2f;
static const float ROT_SPEED = 0.01f;
static const int SAMPLE_PERIOD = 5;
static const char* CONFIG_PATH = "./config.json";
static const char* FONT_PATH = "./assets/arial.ttf";
static const int FONT_SIZE = 12;
static const SDL_Color FOREGROUND_TXT_COL = {255, 255, 255, 255};
static const SDL_Color BACKGROUND_TXT_COL = {0, 0, 0, 0};

int main(int argc, const char** argv) {
    // build scene
    renv::Scene* scene = procedural::generate(CONFIG_PATH);
    std::cout << "Loaded scene" << std::endl;
    // initialize window
    SDL_Init(SDL_INIT_VIDEO);
    TTF_Init();
    bool running = true;
    SDL_Window* window = SDL_CreateWindow("Ray Tracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 
                                                            WIDTH, HEIGHT, SDL_WINDOW_RESIZABLE);
    SDL_Surface* screen = SDL_GetWindowSurface(window);
    SDL_Surface* surface = NULL;
    scene->get_canvas().get_surface(&surface); // link scene near plane to screen
    assert(surface != NULL);
    TTF_Font* font = TTF_OpenFont(FONT_PATH, FONT_SIZE);
    // trap mouse
    bool mouse_trapped = false;
    SDL_SetRelativeMouseMode(SDL_FALSE);
    SDL_ShowCursor(SDL_ENABLE);
    float fps = 0;
    while (running) {
        // create txt
        std::stringstream msg{};
        msg << "FPS: " << fps;
        SDL_Surface* text_surface = TTF_RenderText_Shaded(font, msg.str().c_str(), FOREGROUND_TXT_COL, BACKGROUND_TXT_COL);

        auto from = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < SAMPLE_PERIOD; i++) {
            if (!running) {
                break;
            }

            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                switch (event.type) {
                    case SDL_QUIT: {
                        std::cout << "Exiting" << std::endl;
                        running = false;
                        break;
                    }
                    case SDL_KEYUP: {
                        if (event.key.keysym.sym == SDLK_ESCAPE) {
                            if (mouse_trapped) {
                                SDL_SetRelativeMouseMode(SDL_FALSE);
                                SDL_ShowCursor(SDL_ENABLE);
                            } else {
                                SDL_SetRelativeMouseMode(SDL_TRUE);
                                SDL_ShowCursor(SDL_DISABLE);
                            }

                            mouse_trapped = !mouse_trapped;
                        }
                    }
                    case SDL_KEYDOWN: {
                        if (mouse_trapped) {
                            SDL_KeyboardEvent key_event = event.key;
                            bool key_down = key_event.type == SDL_KEYDOWN;
                            switch (key_event.keysym.sym) {
                                case SDLK_w: {
                                    scene->get_camera().translate(rmath::Vec3<float>{0, 0, MOVE_SPEED});
                                    break;
                                }
                                case SDLK_s: {
                                    scene->get_camera().translate(rmath::Vec3<float>{0, 0, -MOVE_SPEED});
                                    break;
                                }
                                case SDLK_a: {
                                    scene->get_camera().translate(rmath::Vec3<float>{-MOVE_SPEED, 0, 0});
                                    break;
                                }
                                case SDLK_d: {
                                    scene->get_camera().translate(rmath::Vec3<float>{MOVE_SPEED, 0, 0});
                                    break;
                                }
                                case SDLK_ESCAPE: {
                                    break;
                                }
                            }
                        }
                        break;
                    }
                    case SDL_MOUSEMOTION: {
                        if (mouse_trapped) {
                            SDL_MouseMotionEvent mouse_event = event.motion;
                            renv::Camera& cam = scene->get_camera(); 
                            rmath::Vec<float, 2> rel_mot = rmath::Vec<float, 2>({(float) mouse_event.xrel, (float) mouse_event.yrel}).normalized();
                            rmath::Vec3<float> global_mot = rel_mot[0] * cam.right().direction() + rel_mot[1] * cam.up().direction();
                            rmath::Quat<float> rot = rmath::Quat<float>(cam.up().direction(), ROT_SPEED * rel_mot[0]) 
                                        * rmath::Quat<float>(cam.right().direction(), ROT_SPEED * rel_mot[1]);
                            cam.rotate(rot);
                        }
                        break;
                    }
                    case SDL_MOUSEBUTTONDOWN: {
                        if (!mouse_trapped && event.button.button == SDL_BUTTON_LEFT) {
                            std::cout << "shooting debug ray at " << event.button.x << ", " << event.button.y << std::endl;
                            rtracer::debug_cast(scene, event.button.x, event.button.y);
                        }
                        break;
                    }
                }
            }
            SDL_FillRect(screen, NULL, 0x0);
            rtracer::update_scene(scene);
            SDL_BlitSurface(surface, NULL, screen, NULL);
            SDL_BlitSurface(text_surface, NULL, screen, NULL);
            SDL_UpdateWindowSurface(window);
        }
        auto to = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = to - from;
        fps = (double) SAMPLE_PERIOD / elapsed.count();
        SDL_FreeSurface(text_surface);
    }
    SDL_FreeSurface(surface);
    SDL_DestroyWindow(window);
    TTF_CloseFont(font);
    TTF_Quit();
    SDL_Quit();
    return 0;
}