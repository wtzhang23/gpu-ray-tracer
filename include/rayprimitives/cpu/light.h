#ifndef RAYPRIMITIVES_CPU_LIGHT_H
#define RAYPRIMITIVES_CPU_LIGHT_H
#include "raymath/linear.h"
#include "raymath/geometry.h"

namespace renv {
namespace cpu {

class Scene;

}
}

namespace rprimitives {
namespace cpu {

class Light {
protected:
    rmath::Vec4<float> color;
    rmath::Vec4<float> attenuate(const rmath::Ray<float>& to_light, float max_t, renv::cpu::Scene* scene) const;
public:
    Light(): color({1.0f, 1.0f, 1.0f, 1.0f}){}

    void set_color(const rmath::Vec4<float>& col) {
        color = col;
    }

    virtual rmath::Vec4<float> shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::cpu::Scene* scene) const {
        return rmath::Vec4<float>();
    }
};

class PointLight: public Light {
protected:
    rmath::Vec3<float> pos;
public:
    PointLight(): Light(), pos() {}

    void set_pos(const rmath::Vec3<float>& pos) {
        this->pos = pos;
    }

    virtual rmath::Vec4<float> shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::cpu::Scene* scene) const override;
};

class DirLight: public Light {
protected:
    rmath::Vec3<float> dir;
public:
    DirLight(): Light(), dir({0.0f, -1.0f, 0.0f}) {}

    void set_shine_dir(const rmath::Vec3<float>& dir) {
        this->dir = dir.normalized();
    }

    virtual rmath::Vec4<float> shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::cpu::Scene* scene) const override;
};

}
}

#endif