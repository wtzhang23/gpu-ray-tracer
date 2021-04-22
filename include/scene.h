#ifndef SCENE_H
#define SCENE_H

#include "canvas.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace scene {
class Scene {
private:
    canvas::Canvas canvas;
public:
    Scene(canvas::Canvas& canvas): canvas(canvas){}
    CUDA_HOSTDEV
    Scene(const Scene&) = default;

    CUDA_HOSTDEV
    canvas::Canvas& get_canvas() {
        return canvas;
    }
};
}
#endif