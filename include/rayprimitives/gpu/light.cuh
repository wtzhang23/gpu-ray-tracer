#ifndef RAYPRIMITIVES_GPU_LIGHT_CUH
#define RAYPRIMITIVES_GPU_LIGHT_CUH
#include "raymath/linear.h"
#include "raymath/geometry.h"

namespace renv {
namespace gpu {

class Scene;

}
}

namespace rprimitives {
namespace gpu {

class Light {
protected:
    rmath::Vec4<float> color;
    __device__
    rmath::Vec4<float> attenuate(const rmath::Ray<float>& to_light, float max_t, renv::gpu::Scene* scene) const;
public:
    __device__
    Light(): color({1.0f, 1.0f, 1.0f, 1.0f}){}
    
    __device__
    void set_color(const rmath::Vec4<float>& col) {
        color = col;
    }

    __device__
    virtual rmath::Vec4<float> shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::gpu::Scene* scene) const {
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
    void set_pos(const rmath::Vec3<float>& pos) {
        this->pos = pos;
    }

    __device__
    virtual rmath::Vec4<float> shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::gpu::Scene* scene) const override;
};

class DirLight: public Light {
protected:
    rmath::Vec3<float> dir;
public:
    __device__
    DirLight(): Light(), dir({0.0f, -1.0f, 0.0f}) {}

    __device__
    void set_shine_dir(const rmath::Vec3<float>& dir) {
        this->dir = dir.normalized();
    }

    __device__
    virtual rmath::Vec4<float> shine(const rmath::Vec3<float>& hit_pos, rmath::Vec3<float>& dir_to_light, renv::gpu::Scene* scene) const override;
};

}
}

#endif