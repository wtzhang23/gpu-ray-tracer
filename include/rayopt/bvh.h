#ifndef BVH_H
#define BVH_H

#include "raymath/linear.h"
#include "raymath/geometry.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace renv {
class Scene;
}

namespace ropt {
class BoundingBox;

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
    renv::Scene* scene;
    int node_idx;
    int n_int;
    
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
    CUDA_HOSTDEV
    void traverse_up(int max_time);
    CUDA_HOSTDEV
    bool traverse_down(int max_time);
public:
    CUDA_HOSTDEV
    BVHIterator(const rmath::Ray<float>& r, float max_time, renv::Scene* scene);
    CUDA_HOSTDEV
    void next(float max_time);
    CUDA_HOSTDEV
    int current() const;
    
    CUDA_HOSTDEV
    BoundingBox cur_bounding_box() const;

    CUDA_HOSTDEV
    int n_intersections() const {
        return n_int;
    }

    CUDA_HOSTDEV
    int max_intersections() const {
        return bvh.n_objs * 2 - 1;
    }
};
}

#endif