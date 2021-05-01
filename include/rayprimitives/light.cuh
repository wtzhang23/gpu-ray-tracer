#ifndef LIGHT_CUH
#define LIGHT_CUH
#include "raymath/linear.h"
#include "rayenv/scene.h"

namespace rprimitives {
class Light {
protected:
    rmath::Vec4<float> color;
    __device__
    rmath::Vec4<float> attenuate(const rmath::Ray<float>& to_light, float max_t, renv::Scene* scene) const;
public:
    __device__
    Light(): color(rmath::Vec4<float>({1.0f, 1.0f, 1.0f, 1.0f})){}
    
    __device__
    void set_color(rmath::Vec4<float>& col) {
        color = col;
    }

    __device__
    virtual rmath::Vec4<float> shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::Scene* scene) const {
        return rmath::Vec4<float>();
    }
};

class PointLight: public Light {
protected:
    rmath::Vec3<float> pos;
public:
    __device__
    PointLight(): Light(), pos() {}

    __device__
    void set_pos(rmath::Vec3<float>& pos) {
        this->pos = pos;
    }

    __device__
    virtual rmath::Vec4<float> shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::Scene* scene) const override;
};

class DirLight: public Light {
protected:
    rmath::Vec3<float> dir;
public:
    __device__
    DirLight(): Light(), dir({0.0f, -1.0f, 0.0f}) {}

    __device__
    void set_shine_dir(rmath::Vec3<float>& dir) {
        this->dir = dir.normalized();
    }

    __device__
    virtual rmath::Vec4<float> shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::Scene* scene) const override;
};
}

#endif