#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "rayenv/scene.h"

namespace rtracer {
    template <typename T>
    void update_scene(renv::Scene<T>& scene);
}

#endif