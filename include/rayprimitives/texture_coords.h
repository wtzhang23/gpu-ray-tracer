#ifndef RAYPRIMITIVES_TEXTURE_COORDS_H
#define RAYPRIMITIVES_TEXTURE_COORDS_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace rprimitives {

struct TextureCoords {
    int texture_x;
    int texture_y;
    rmath::Vec<float, 2> u;
    rmath::Vec<float, 2> v;
    bool degenerate;
    CUDA_HOSTDEV
    TextureCoords(int texture_x, int texture_y, rmath::Vec<float, 2> u, rmath::Vec<float, 2> v): 
                    texture_x(texture_x), texture_y(texture_y), u(u), v(v), degenerate(false) {}

    CUDA_HOSTDEV
    TextureCoords(): texture_x(0), texture_y(0), u(), v(), degenerate(true){}

    CUDA_HOSTDEV
    bool is_degenerate() const {
        return degenerate;
    }
};

}

#endif