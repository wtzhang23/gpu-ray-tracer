#include <iostream>
#include <SDL2/SDL.h>
#include <thread>
#include <atomic>
#include <mutex>
#include "raytracer.h"
#include "rayenv/canvas.h"
#include "raymath/geometry.h"
#include "raymath/linear.h"
#include "rayenv/scene.h"
#include "scene_builder.h"

static const int WIDTH = 640;
static const int HEIGHT = 480;
static const float MOVE_SPEED = 0.2f;
static const float ROT_SPEED = 0.01f;
static const char* ATLAS_PATH = "../assets/sus.png";

int main(int argc, const char** argv) {
    // build scene
    renv::Canvas canvas{WIDTH, HEIGHT};
    renv::Camera cam{M_PI / 4, 200, canvas};
    rtracer::SceneBuilder scene_builder{std::string(ATLAS_PATH)};
    rprimitives::Material mat{};
    mat.set_Ke({0.0f, 0.0f, 0.0f, 1.0f});
    mat.set_Kd({1.0f, 0.0f, 0.0f, 1.0f});
    mat.set_Ka({0.2f, 0.2f, 0.2f, 1.0f});
    mat.set_Ks({1.0f, 1.0f, 1.0f, 1.0f});
    mat.set_Kt({0.2f, 0.2f, 0.2f, 1.0f});
    mat.set_alpha(.8f);
    scene_builder.build_cube(1.0f, 
                rmath::Vec3<float>({0.0f, 0.0f, 5.0f}),
                rmath::Quat<float>::identity(),
                rprimitives::Shade(rmath::Vec4<float>({0.0f, 1.0f, 0.0f, 1.0f})),
                mat
    );
    scene_builder.add_directional_light({0.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f, 1.0f});
    scene_builder.set_ambience({0.0f, 0.0f, 1.0f, 1.0f});
    scene_builder.set_recurse_depth(1);
    renv::Scene* scene = scene_builder.build_scene(canvas, cam);
    std::cout << "Loaded scene" << std::endl;
    // initialize window
    SDL_Init(SDL_INIT_VIDEO);
    bool running = true;
    SDL_Window* window = SDL_CreateWindow("Ray Tracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 
                                                            WIDTH, HEIGHT, SDL_WINDOW_RESIZABLE);
    SDL_Surface* screen = SDL_GetWindowSurface(window);
    SDL_Surface* surface = NULL;
    scene->get_canvas().get_surface(&surface); // link scene near plane to screen
    assert(surface != NULL);
    // trap mouse
    bool mouse_trapped = false;
    SDL_SetRelativeMouseMode(SDL_FALSE);
    SDL_ShowCursor(SDL_ENABLE);
    while (running) {
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
                        rmath::Vec<float, 2> rel_mot = rmath::Vec<float, 2>({(float) mouse_event.xrel, (float) -mouse_event.yrel}).normalized();
                        rmath::Vec3<float> global_mot = rel_mot[0] * cam.right().direction() + rel_mot[1] * cam.up().direction();
                        rmath::Vec3<float> axis = rmath::cross(global_mot, cam.forward().direction()).normalized();
                        rmath::Quat<float> rot = rmath::Quat<float>(axis, -ROT_SPEED);
                        cam.rotate(rot);
                    }
                    break;
                }
            }
        }
        SDL_FillRect(screen, NULL, 0x0);
        rtracer::update_scene(scene);
        int rv = SDL_BlitSurface(surface, NULL, screen, NULL);
        assert(rv == 0);
        rv = SDL_UpdateWindowSurface(window);
        assert(rv == 0);
    }
    SDL_FreeSurface(surface);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}