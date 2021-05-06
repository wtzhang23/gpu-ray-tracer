#include "rayopt/gpu/bvh.h"
#include "rayopt/bounding_box.h"
#include "rayopt/z_order.h"
#include "raymath/linear.h"
#include "rayenv/gpu/scene.h"
#include <thrust/sort.h>

namespace ropt {
namespace gpu {

__global__
void gen_numbers(int* arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        arr[i] = i;
    }
}

__global__
void gen_morton(unsigned long* codes, BoundingBox* boxes, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        if (boxes[i].is_degenerate()) {
            codes[i] = ULONG_MAX;
        } else {
            rmath::Vec3<float> center = -boxes[i].center(); // negate since first bit indicates pos/neg
            codes[i] = ropt::z_order(center);
        }
    }
}

__global__
void reorder(int* ordering, BoundingBox* from_boxes, BoundingBox* to_boxes, int n_objs) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n_objs; i += stride) {
        to_boxes[i] = from_boxes[ordering[i]];
    }    
}

__global__
void build_bvh_layer(BoundingBox* boxes, int batch_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    BoundingBox* from = boxes;
    while (batch_size >= 2) {
        int offset = idx * batch_size;
        int next_offset = offset >> 1;
        int next_addr = blockDim.x * gridDim.x * batch_size;
        assert(offset < next_addr);
        BoundingBox* to = &from[next_addr];
        for (int i = 0; i < (batch_size >> 1); i++) {
            BoundingBox left = from[offset + 2 * i];
            BoundingBox right = from[offset + 2 * i + 1];
            to[next_offset + i] = ropt::merge(left, right);
        }
        from = to;
        batch_size >>= 1;
    }
}

const int BVH_THREADS = 512;
__host__
void build_bvh(BoundingBox* flattened_tree, int n_boxes) {
    if (n_boxes >= BVH_THREADS * 2) {
        int n_blocks = n_boxes / (2 * BVH_THREADS);
        build_bvh_layer<<<n_blocks, BVH_THREADS>>>(flattened_tree, 2); // reduce one layer
        build_bvh(flattened_tree + (n_blocks * BVH_THREADS * 2), n_boxes / 2);
    } else {
        build_bvh_layer<<<1, 1>>>(flattened_tree, n_boxes);
    }
}
BVH::BVH(BoundingBox* org_boxes, int n_org_objs): n_objs(n_org_objs) {
    int n_blocks = (n_org_objs + BVH_THREADS - 1) / BVH_THREADS;
    unsigned long* codes;
    BoundingBox* flattened_tree;
    int rv = cudaMalloc(&ordering, n_org_objs * sizeof(int));
    assert(rv == 0);
    rv = cudaMalloc(&codes, n_org_objs * sizeof(unsigned long));
    assert(rv == 0);
    rv = cudaMalloc(&flattened_tree, 2 * n_org_objs * sizeof(BoundingBox));
    assert(rv == 0);
    gen_morton<<<n_blocks, BVH_THREADS>>>(codes, org_boxes, n_org_objs);
    gen_numbers<<<n_blocks, BVH_THREADS>>>(ordering, n_org_objs);
    thrust::sort_by_key(thrust::device, codes, codes + n_org_objs, ordering);
    reorder<<<n_blocks, BVH_THREADS>>>(ordering, org_boxes, flattened_tree, n_objs);
    build_bvh(flattened_tree, n_org_objs);
    this->boxes = flattened_tree;
    cudaFree(codes);
}

void BVH::free(BVH& b) {
    cudaFree(b.boxes);
    cudaFree(b.ordering);
}
CUDA_HOSTDEV
void BVHIterator::traverse_up(int max_time) {
    // go up the tree until first branch
    assert(node_idx >= 1);
    while (true) { 
        while (node_idx % 2 == 1) {
            node_idx = parent();
        }

        if (node_idx == 0) {
            break;
        }

        node_idx = parent();
        assert(node_idx >= 1);
        int rc = right_child();
        assert(rc >= 0);
        int rc_box_idx = BVHIterator::get_box_idx(rc);
        float time;
        n_int++;
        if (bvh.boxes[rc_box_idx].intersects(r, time) && time < max_time) {
            node_idx = rc;
            break;
        }
    }
}
CUDA_HOSTDEV
bool BVHIterator::traverse_down(int max_time) {
    assert(node_idx >= 1);
    while (true) {
        int lc = left_child();
        int rc = right_child();
        int lc_box_idx = get_box_idx(lc);
        int rc_box_idx = get_box_idx(rc);
        if (lc < 0) {
            assert(rc < 0);
            return true;
        }
        float time;
        if (bvh.boxes[lc_box_idx].intersects(r, time) && time < max_time) {
            n_int++;
            node_idx = lc;
        } else if (bvh.boxes[rc_box_idx].intersects(r, time) && time < max_time) {
            n_int += 2;
            node_idx = rc;
        } else {
            n_int += 2;
            return false;
        }
    }
}
CUDA_HOSTDEV
BVHIterator::BVHIterator(const rmath::Ray<float>& r, float max_time, renv::gpu::Scene* scene): 
            bvh(scene->get_bvh()), r(r), node_idx(0), scene(scene), n_int(0) {
    float time;
    BoundingBox& top_box = bvh.boxes[get_box_idx(1)];
    if (top_box.intersects(r, time) && time < max_time) {        
        node_idx = 1;
        while (node_idx > 0) {
            if (traverse_down(max_time)) {
                break;
            } else {
                traverse_up(max_time);
            }
        }
    }
}
CUDA_HOSTDEV
void BVHIterator::next(float max_time) {
    if (node_idx == 0) {
        return;
    }
    while (true) {
        traverse_up(max_time);
        if (node_idx == 0 || traverse_down(max_time)) {
            break;
        }
    }
}
CUDA_HOSTDEV
int BVHIterator::current() const {
    if (node_idx == 0) {
        return -1;
    }
    int box_idx = get_box_idx(node_idx);
    int after_order = bvh.ordering[box_idx];
    return after_order;
}
CUDA_HOSTDEV
BoundingBox BVHIterator::cur_bounding_box() const {
    if (node_idx == 0) {
        return BoundingBox{};
    }
    int box_idx = get_box_idx(node_idx);
    return bvh.boxes[box_idx];
}

}
}