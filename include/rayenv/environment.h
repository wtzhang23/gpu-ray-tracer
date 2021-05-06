#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayenv/canvas.h"
#include "rayenv/camera.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace renv {

class Transformation;

class Environment {
private:
    Canvas canvas;
    Camera cam;
    rmath::Vec3<float> dist_atten;
    rmath::Vec4<float> ambience;
    Transformation* trans;
    int nt;
    int recurse_depth;
    bool debugging;
public:
    Environment(Canvas canvas, Camera camera, Transformation* trans, int n_trans): canvas(canvas), dist_atten(), ambience(), 
                                            cam(camera), trans(trans), nt(n_trans), recurse_depth(0), debugging(false){}
    void set_dist_atten(float const_term, float linear_term, float quad_term) {
        dist_atten = rmath::Vec3<float>({const_term, linear_term, quad_term});
    }

    void set_ambience(rmath::Vec4<float> amb) {
        ambience = amb;
    }

    void set_recurse_depth(int depth) {
        recurse_depth = depth;
    }

    void set_debug_mode(bool mode) {
        debugging = mode;
    }

    CUDA_HOSTDEV
    bool is_debugging() const {
        return debugging;
    }

    CUDA_HOSTDEV
    const rmath::Vec3<float>& get_dist_atten() const {
        return dist_atten;
    }

    CUDA_HOSTDEV
    const rmath::Vec4<float>& get_ambience() const {
        return ambience;
    }

    CUDA_HOSTDEV
    int get_recurse_depth() const {
        return recurse_depth;
    }

    CUDA_HOSTDEV
    Canvas& get_canvas() {
        return canvas;
    }

    CUDA_HOSTDEV
    Camera& get_camera() {
        return cam;
    }

    CUDA_HOSTDEV
    Transformation* get_trans() {
        return trans;
    }

    CUDA_HOSTDEV
    int n_trans() const {
        return nt;
    }
};

}

#endif