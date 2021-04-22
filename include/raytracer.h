#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "scene.h"

namespace raytracer {
    template <typename T>
    void update_scene(scene::Scene<T>& scene);
}

#endif