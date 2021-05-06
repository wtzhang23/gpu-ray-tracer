#ifndef RAYPRIMITIVES_CPU_PHONG_H
#define RAYPRIMITIVES_CPU_PHONG_H

#include "raymath/geometry.h"

namespace renv {
namespace cpu {

class Scene;

}
}

namespace rprimitives {

class Isect;

namespace cpu {

rmath::Vec4<float> illuminate(const rmath::Ray<float>& org_ray, const Isect& i, renv::cpu::Scene* s);

}
}

#endif