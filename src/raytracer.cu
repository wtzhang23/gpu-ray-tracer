#include <thrust/scan.h>
#include "raymath/linear.h"
#include "raytracer.h"
#include "iostream"
#include "rayenv/scene.h"
#include "rayenv/scene.cuh"
#include "rayprimitives/texture.cuh"
#include "gputils/alloc.h"
#include "assets.h"

namespace rtracer {
static const int SQ_WIDTH = 16;

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
            rmath::Vec4<float> c = renv::propagate_ray(scene, r);
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
    rmath::Vec4<float> c = renv::propagate_ray(scene, r);
}

void debug_cast(renv::Scene* scene, int x, int y) {
    scene->set_debug_mode(true);
    debug<<<1, 1>>>(scene, x, y);
    cudaDeviceSynchronize();
    scene->set_debug_mode(false);
}

void update_scene(renv::Scene* scene) {
    dim3 dimBlock(SQ_WIDTH, SQ_WIDTH);
    int grid_dim_x = scene->get_canvas().get_width() / SQ_WIDTH;
    int grid_dim_y = scene->get_canvas().get_height() / SQ_WIDTH;
    dim3 dimGrid(grid_dim_x == 0 ? 1 : grid_dim_x, grid_dim_y == 0 ? 1 : grid_dim_y);
    trace<<<dimGrid, dimBlock>>>(scene);
    int rv = cudaDeviceSynchronize();
    assert(rv == 0);
    renv::Canvas& canvas = scene->get_canvas();
}
}