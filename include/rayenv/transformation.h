#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include "rayprimitives/entity.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace renv {
class Transformation: public rprimitives::Entity {
private:
    int hitable_idx;
public:
    Transformation(int hi): hitable_idx(hi) {}
    CUDA_HOSTDEV
    int get_hitable_idx() const {
        return hitable_idx;
    }
};
}

#endif