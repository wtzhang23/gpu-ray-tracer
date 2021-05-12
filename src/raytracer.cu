#include <thrust/scan.h>
#include <iostream>
#include "raymath/linear.h"
#include "raytracer.h"
#include "rayenv/gpu/scene.h"
#include "rayenv/gpu/scene.cuh"
#include "rayenv/transformation.h"
#include "rayprimitives/gpu/texture.cuh"
#include "rayprimitives/gpu/hitable.cuh"
#include "rayopt/gpu/bvh.h"
#include "rayopt/bounding_box.h"
#include "gputils/alloc.h"
#include "assets.h"

namespace rtracer {
namespace gpu {
__global__
void trace(renv::gpu::Scene* scene) {
    renv::Environment& env = scene->get_environment();
    renv::Canvas& canvas = env.get_canvas();
    renv::Camera& cam = env.get_camera();
    rprimitives::gpu::Texture& atlas = scene->get_atlas();
    
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    int width = canvas.get_width();
    int height = canvas.get_height();
    for (int i = x; i < width; i += stride_x) {
        for (int j = y; j < height; j += stride_y) {
            rmath::Ray<float> r = cam.at(i, j);
            float time = INFINITY;
            rprimitives::Isect isect{time};
            rmath::Vec4<float> c = renv::gpu::propagate_ray(scene, r, isect);
            canvas.set_color(i, j, renv::Color(c[0] > 1.0f ? 1.0f : c[0], 
                                            c[1] > 1.0f ? 1.0f : c[1], 
                                            c[2] > 1.0f ? 1.0f : c[2], 
                                            c[3] > 1.0f ? 1.0f : c[3]));
        }
    }
}

__global__
void debug(renv::gpu::Scene* scene, int x, int y) {
    renv::Camera& cam = scene->get_environment().get_camera();
    rmath::Ray<float> r = cam.at(x, y);
    float time;
    rprimitives::Isect isect{time};
    rmath::Vec4<float> c = renv::gpu::propagate_ray(scene, r, isect);
}

__global__
void create_boxes(renv::gpu::Scene* scene, ropt::BoundingBox* boxes, int padded_n_boxes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    renv::Environment& env = scene->get_environment();
    renv::Transformation* trans = env.get_trans();
    rprimitives::gpu::Hitable** hitables = scene->get_hitables();
    int n_trans = env.n_trans();
    int n_hitables = scene->n_hitables();

    for (int i = idx; i < padded_n_boxes; i += stride) {
        if (i < n_trans) {
            renv::Transformation& t = trans[i];
            assert(t.get_hitable_idx() < n_hitables);
            rprimitives::gpu::Hitable& h = *hitables[t.get_hitable_idx()];
            boxes[i] = ropt::from_local(h.compute_bounding_box(scene), t);
        } else {
            boxes[i] = ropt::BoundingBox(); // empty box
        }
    }
}

const int BOX_THREADS = 512;
ropt::gpu::BVH create_bvh(renv::gpu::Scene* scene) {
    // generate bounding boxes for bvh usage
    int org_size = scene->get_environment().n_trans();
    int padded_size = 1 << ((int) ceil(log2(org_size))); // ensure # of boxes is a power of 2
    ropt::BoundingBox* org_boxes;
    int rv = cudaMalloc(&org_boxes, sizeof(ropt::BoundingBox) * padded_size);
    assert(rv == 0);
    int n_blocks = (padded_size + BOX_THREADS - 1) / BOX_THREADS;
    create_boxes<<<n_blocks, BOX_THREADS>>>(scene, org_boxes, padded_size);
    ropt::gpu::BVH bvh{org_boxes, padded_size};
    cudaFree(org_boxes);
    return bvh;
}

void debug_cast(renv::gpu::Scene* scene, int x, int y) {
    ropt::gpu::BVH bvh = create_bvh(scene);
    scene->get_environment().set_debug_mode(true);
    scene->set_bvh(bvh);
    debug<<<1, 1>>>(scene, x, y);
    cudaDeviceSynchronize();
    scene->set_bvh(ropt::gpu::BVH{});
    scene->get_environment().set_debug_mode(false);
    ropt::gpu::BVH::free(bvh);
}

void update_scene(renv::gpu::Scene* scene, int kernel_dim, bool optimize) {
    ropt::gpu::BVH bvh;
    if (optimize) {
        bvh = create_bvh(scene);
        scene->set_bvh(bvh);
    }
    dim3 dimBlock(kernel_dim, kernel_dim);
    renv::Canvas& canvas = scene->get_environment().get_canvas();
    int grid_dim_x = canvas.get_width() / kernel_dim;
    int grid_dim_y = canvas.get_height() / kernel_dim;
    dim3 dimGrid(grid_dim_x == 0 ? 1 : grid_dim_x, grid_dim_y == 0 ? 1 : grid_dim_y);
    trace<<<dimGrid, dimBlock>>>(scene);
    int rv = cudaDeviceSynchronize();
    assert(rv == 0);
    scene->set_bvh(ropt::gpu::BVH{}); // unset for safety reasons
    if (optimize) {
        ropt::gpu::BVH::free(bvh);
    }
}
}
}