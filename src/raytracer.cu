#include <thrust/scan.h>
#include "raymath/linear.h"
#include "raytracer.h"
#include "iostream"
#include "rayenv/scene.h"
#include "rayenv/scene.cuh"
#include "rayprimitives/texture.cuh"
#include "gputils/alloc.h"
#include "assets.h"
#include "rayopt/bvh.h"
#include "rayopt/bounding_box.h"
#include "rayprimitives/hitable.cuh"

namespace rtracer {
constexpr int SQ_WIDTH = 22;

__global__
void trace(renv::Scene* scene) {
    renv::Canvas& canvas = scene->get_canvas();
    renv::Camera& cam = scene->get_camera();
    rprimitives::Texture& atlas = scene->get_atlas();
    
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int i = x; i < canvas.get_width(); i += stride_x) {
        for (int j = y; j < canvas.get_height(); j += stride_y) {
            rmath::Ray<float> r = cam.at(i, j);
            float time = INFINITY;
            rprimitives::Isect isect{time};
            rmath::Vec4<float> c = renv::propagate_ray(scene, r, isect);
            canvas.set_color(i, j, renv::Color(c[0] > 1.0f ? 1.0f : c[0], 
                                            c[1] > 1.0f ? 1.0f : c[1], 
                                            c[2] > 1.0f ? 1.0f : c[2], 
                                            c[3] > 1.0f ? 1.0f : c[3]));
        }
    }
}

__global__
void debug(renv::Scene* scene, int x, int y) {
    renv::Camera& cam = scene->get_camera();
    rmath::Ray<float> r = cam.at(x, y);
    float time;
    rprimitives::Isect isect{time};
    rmath::Vec4<float> c = renv::propagate_ray(scene, r, isect);
}

__global__
void create_boxes(renv::Scene* scene, ropt::BoundingBox* boxes, int padded_n_boxes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    renv::Transformation* trans = scene->get_trans();
    rprimitives::Hitable** hitables = scene->get_hitables();
    int n_trans = scene->n_trans();
    int n_hitables = scene->n_hitables();
    for (int i = idx; i < padded_n_boxes; i += stride) {
        if (i < n_trans) {
            renv::Transformation& t = trans[i];
            assert(t.get_hitable_idx() < n_hitables);
            rprimitives::Hitable& h = *hitables[t.get_hitable_idx()];
            boxes[i] = ropt::from_local(h.compute_bounding_box(scene), t);
        } else {
            boxes[i] = ropt::BoundingBox();
        }
    }
}

const int BOX_THREADS = 512;
ropt::BVH update_bvh(renv::Scene* scene) {
    // generate bounding boxes for bvh usage
    int org_size = scene->n_trans();
    int padded_size = 1 << ((int) ceil(log2(org_size)));
    ropt::BoundingBox* org_boxes;
    int rv = cudaMalloc(&org_boxes, sizeof(ropt::BoundingBox) * padded_size);
    assert(rv == 0);
    int n_blocks = (padded_size + BOX_THREADS - 1) / BOX_THREADS;
    create_boxes<<<n_blocks, BOX_THREADS>>>(scene, org_boxes, padded_size);
    ropt::BVH bvh{org_boxes, padded_size};
    cudaFree(org_boxes);
    return bvh;
}

void debug_cast(renv::Scene* scene, int x, int y) {
    ropt::BVH bvh = update_bvh(scene);
    scene->set_debug_mode(true);
    scene->set_bvh(bvh);
    debug<<<1, 1>>>(scene, x, y);
    cudaDeviceSynchronize();
    scene->set_bvh(ropt::BVH{});
    scene->set_debug_mode(false);
    ropt::BVH::free(bvh);
}

void update_scene(renv::Scene* scene) {
    ropt::BVH bvh = update_bvh(scene);
    scene->set_bvh(bvh);
    dim3 dimBlock(SQ_WIDTH, SQ_WIDTH);
    int grid_dim_x = scene->get_canvas().get_width() / SQ_WIDTH;
    int grid_dim_y = scene->get_canvas().get_height() / SQ_WIDTH;
    dim3 dimGrid(grid_dim_x == 0 ? 1 : grid_dim_x, grid_dim_y == 0 ? 1 : grid_dim_y);
    trace<<<dimGrid, dimBlock>>>(scene);
    int rv = cudaDeviceSynchronize();
    assert(rv == 0);
    scene->set_bvh(ropt::BVH{}); // unset for safety reasons
    ropt::BVH::free(bvh);
}
}