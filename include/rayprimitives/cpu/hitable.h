#ifndef RAYPRIMITIVES_CPU_HITABLE_H
#define RAYPRIMITIVES_CPU_HITABLE_H

#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayopt/bounding_box.h"
#include "rayprimitives/entity.h"
#include "rayprimitives/isect.h"

namespace renv {
namespace cpu {

class Scene;

}
}

namespace rprimitives {
namespace cpu {

class Boxed {
public:
    virtual ropt::BoundingBox compute_bounding_box(renv::cpu::Scene* scene) const {
        return ropt::BoundingBox{};
    }
};

class Hitable: public Entity, public Boxed {
public:
    Hitable(): Entity() {}
    Hitable(rmath::Vec3<float> position, rmath::Quat<float> orientation): Entity(position, orientation) {}
    Hitable(Entity* entity): Entity(entity) {}
    virtual ~Hitable() {};
    
    virtual bool hit_local(const rmath::Ray<float>& local_ray, renv::cpu::Scene* scene, Isect& isect) const {
        return false;
    };
    
    bool hit(const rmath::Ray<float>& ray, renv::cpu::Scene* scene, Isect& isect) const;
};

}
}

#endif