#include "rayenv/cpu/scene.h"
#include "rayenv/transformation.h"
#include "rayprimitives/cpu/hitable.h"
#include "rayprimitives/cpu/light.h"

namespace renv {

namespace cpu {

Scene::~Scene() {
    for (rprimitives::cpu::Hitable* h : hitables) {
        delete h;
    }

    for (rprimitives::cpu::Light* l : lights) {
        delete l;
    }

    if (env.get_trans() != NULL) {
        delete[] env.get_trans();
    }
}

Scene::Scene(Scene&& scene): env(scene.env), atlas(scene.atlas),
                    hitables(scene.hitables), lights(scene.lights), buffer(scene.buffer){
    scene.hitables.clear();
    scene.lights.clear();
    env.clear_transformations();
} 

Scene& Scene::operator=(Scene&& scene) {
    for (rprimitives::cpu::Hitable* h : hitables) {
        delete h;
    }

    for (rprimitives::cpu::Light* l : lights) {
        delete l;
    }
    hitables = scene.hitables;
    lights = scene.lights;
    scene.hitables.clear();
    scene.lights.clear();
    env.clear_transformations();
    return *this;
}

}

}