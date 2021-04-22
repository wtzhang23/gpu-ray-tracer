#include "linear.h"
#include "raytracer.h"
#include "iostream"

namespace raytracer {
    template <typename T>
    __global__
    void color_green(scene::Scene<T> scene) {
        canvas::Canvas& canvas = scene.get_canvas();
        int height = canvas.get_height();
        int width = canvas.get_width();
        
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = idx; i < height * width; i += stride) {
            int row = i / width;
            int col = i % width;
            canvas.set_color(row, col, canvas::Color(0, 1.0, 0));
        }
    }

    template <typename T>
    void update_scene(scene::Scene<T>& scene) {
        color_green<<<1024, 1024>>>(scene);
        int rv = cudaDeviceSynchronize();
        assert(rv == 0);
        canvas::Canvas& canvas = scene.get_canvas();
    }

    template void update_scene<double>(scene::Scene<double>& scene);
    template void update_scene<float>(scene::Scene<float>& scene);
}