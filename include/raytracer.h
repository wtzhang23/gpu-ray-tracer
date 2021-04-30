#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "rayenv/scene.h"

namespace rtracer {
    renv::Scene* build_scene(int width, int height);
    void update_scene(renv::Scene* scene);
}

#endif