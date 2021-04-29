#include "raymath/linear.h"
#include "raytracer.h"
#include "iostream"
#include "rayprimitives/texture.cuh"

namespace rtracer {
    __global__
    void trace(renv::Scene scene) {
        renv::Canvas& canvas = scene.get_canvas();
        rprimitives::Texture& atlas = scene.get_atlas();
        
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int stride_x = blockDim.x * gridDim.x;
        int stride_y = blockDim.y * gridDim.y;

        for (int i = x; i < canvas.get_width(); i += stride_x) {
            for (int j = y; j < canvas.get_height(); j += stride_y) {
                rmath::Vec4<float> norm_col = get_color_from_texture(atlas, i, j);
                canvas.set_color(i, j, renv::Color(norm_col[0], norm_col[1], norm_col[2], norm_col[3]));
            }
        }
    }

    void update_scene(renv::Scene& scene) {
        dim3 dimBlock(32, 32);
        int grid_dim_x = scene.get_canvas().get_width() / 32;
        int grid_dim_y = scene.get_canvas().get_height() / 32;
        dim3 dimGrid(grid_dim_x == 0 ? 1 : grid_dim_x, grid_dim_y == 0 ? 1 : grid_dim_y);
        trace<<<dimGrid, dimBlock>>>(scene);
        int rv = cudaDeviceSynchronize();
        assert(rv == 0);
        renv::Canvas& canvas = scene.get_canvas();
    }
}