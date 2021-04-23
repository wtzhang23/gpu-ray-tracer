#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "rayenv/scene.h"

namespace raytracer {
    template <typename T>
    void update_scene(renv::Scene<T>& scene);
}

#endif