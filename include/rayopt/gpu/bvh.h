#ifndef RAYOPT_GPU_BVH_H
#define RAYOPT_GPU_BVH_H

#include "raymath/linear.h"
#include "raymath/geometry.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace renv {
namespace gpu {

class Scene;

}
}

namespace ropt {
class BoundingBox;

namespace gpu {

class BVH {
private:
    BoundingBox* boxes;
    int* ordering;
    int n_objs;
public:
    BVH(): boxes(NULL), ordering(NULL), n_objs(0) {}
    BVH(BoundingBox* org_boxes, int n_org_objs);
    CUDA_HOSTDEV
    bool empty() const {
        return n_objs == 0;
    }

    static void free(BVH& b);
    friend class BVHIterator;
};

class BVHIterator {
private:
    const BVH& bvh;
    const rmath::Ray<float>& r;
    renv::gpu::Scene* scene;
    int node_idx;
    
    CUDA_HOSTDEV
    int get_box_idx(int i) const {
        return bvh.n_objs * 2 - i - 1;
    }
    CUDA_HOSTDEV
    int left_child() const {
        if (node_idx * 2 >= bvh.n_objs * 2 - 1) {
            return -1;
        }
        return node_idx * 2;
    }
    CUDA_HOSTDEV
    int right_child() const {
        if (node_idx * 2 >= bvh.n_objs * 2 - 1) {
            return -1;
        }
        return node_idx * 2 + 1;
    }
    CUDA_HOSTDEV
    int parent() const {
        return node_idx / 2;
    }
public:
    CUDA_HOSTDEV
    BVHIterator(const rmath::Ray<float>& r, renv::gpu::Scene* scene);
    
    CUDA_HOSTDEV
    bool at_child() const {
        return node_idx * 2 >= bvh.n_objs * 2 - 1;
    }

    CUDA_HOSTDEV
    bool intersects_node() const;
    
    CUDA_HOSTDEV
    int current() const;
    
    CUDA_HOSTDEV
    void step_next();

    CUDA_HOSTDEV
    void step_up();

    CUDA_HOSTDEV
    BoundingBox cur_bounding_box() const;

    CUDA_HOSTDEV
    bool running() const {
        return node_idx >= 1;
    }
};

}
}

#endif