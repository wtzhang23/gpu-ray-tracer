#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <chrono>
#include <sstream>
#include <iostream>
#include <cuda.h>
#include <cxxopts.hpp>
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
static const char* FONT_PATH = "./assets/arial.ttf";
static const int FONT_SIZE = 12;
static const SDL_Color FOREGROUND_TXT_COL = {255, 255, 255, 255};
static const SDL_Color BACKGROUND_TXT_COL = {0, 0, 0, 0};
static const char* DEFAULT_KERNEL_DIM = "16";

void open_window(renv::Scene* scene, int kernel_dim, bool disable_opt);
void bench(renv::Scene* scene, int kernel_dim, bool disable_opt);

int main(int argc, const char** argv) {
    cxxopts::Options options("Ray Tracer", "A gpu-accelerated ray tracer.");
    options.add_options()
        ("c,config", "Configuration file (json)", cxxopts::value<std::string>())
        ("b,bench", "Benchmark mode", cxxopts::value<bool>()->default_value("false"))
        ("r,unoptimize", "Disable optimizing data structures", cxxopts::value<bool>()->default_value("false"))
        ("d,dim", "Kernel dimension", cxxopts::value<int>()->default_value(DEFAULT_KERNEL_DIM));
    auto opt_res = options.parse(argc, argv);
    std::string config_path = opt_res["config"].as<std::string>();
    int kernel_dim = opt_res["dim"].as<int>();
    bool b = opt_res["bench"].as<bool>();
    bool r = opt_res["unoptimize"].as<bool>();

    // build scene
    renv::Scene* scene = procedural::generate(config_path);
    std::cout << "Loaded scene" << std::endl;
    if (b) {
        bench(scene, kernel_dim, r);
    } else {
        open_window(scene, kernel_dim, r);
    }
    return 0;
}

void open_window(renv::Scene* scene, int kernel_dim, bool unoptimize) {
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
    bool draw_txt = false;
    bool mouse_trapped = false;
    SDL_SetRelativeMouseMode(SDL_FALSE);
    SDL_ShowCursor(SDL_ENABLE);
    float fps = 0;
    while (running) {
        // create txt
        std::stringstream msg{};
        msg << "FPS: " << fps << " Locked: " << mouse_trapped;
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
                        switch (event.key.keysym.sym) {
                            case SDLK_ESCAPE: {
                                if (mouse_trapped) {
                                    SDL_SetRelativeMouseMode(SDL_FALSE);
                                    SDL_ShowCursor(SDL_ENABLE);
                                } else {
                                    SDL_SetRelativeMouseMode(SDL_TRUE);
                                    SDL_ShowCursor(SDL_DISABLE);
                                }
                                mouse_trapped = !mouse_trapped;
                                break;
                            }
                            case SDLK_F1: {
                                draw_txt = !draw_txt;
                                break;
                            }
                        }
                        break;
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
            rtracer::update_scene(scene, kernel_dim, !unoptimize);
            SDL_BlitSurface(surface, NULL, screen, NULL);
            if (draw_txt) {
                SDL_BlitSurface(text_surface, NULL, screen, NULL);
            }
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
}

void bench(renv::Scene* scene, int kernel_dim, bool unoptimize) {
    auto from = std::chrono::high_resolution_clock::now();
    rtracer::update_scene(scene, kernel_dim, !unoptimize);
    auto to = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = to - from;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms" << std::endl;
}