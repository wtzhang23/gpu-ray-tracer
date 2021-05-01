#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "rayenv/scene.h"

namespace rtracer {
    void update_scene(renv::Scene* scene);
    void debug_cast(renv::Scene* scene, int x, int y);
}

#endif