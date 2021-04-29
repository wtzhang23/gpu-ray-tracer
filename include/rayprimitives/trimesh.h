#ifndef TRIMESH_H
#define TRIMESH_H

#include <vector>
#include <array>
#include <memory>
#include "rayprimitives/entity.h"
#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayprimitives/material.h"
#include "rayprimitives/texture.h"
#include "rayprimitives/entity.h"
#include "gputils/alloc.h"
#include "gputils/flat_vec.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace rprimitives {
class VertexBuffer {
private:
    gputils::TextureBuffer4D<float> v;
    gputils::TextureBuffer4D<float> n;
public:
    VertexBuffer(std::vector<rmath::Vec3<float>>& vertices, std::vector<rmath::Vec3<float>>& normals);
    
    CUDA_HOSTDEV
    const gputils::TextureBuffer4D<float>& get_vertices() const {
        return v;
    }

    CUDA_HOSTDEV
    const gputils::TextureBuffer4D<float>& get_normals() const {
        return n;
    }
};

class TriInner: public Entity {
private:
    rmath::Vec3<int> indices;
    union Shade {
        struct TextData {
            int texture_x;
            int texture_y;
            int texture_width;
            int texture_height;
        } text_data;
        rmath::Vec4<float> col;
        Shade(rmath::Vec4<float> col): col(col) {}
        Shade(int texture_x, int texture_y, int texture_width, int texture_height): text_data{texture_x, texture_y, texture_width, texture_height}{}
    } shading;
    Material mat;
    bool use_texture;
public:
    TriInner(rmath::Vec3<int> indices, rmath::Vec4<float> col, Material mat): indices(indices), shading(col), mat(mat), use_texture(false){}
    TriInner(rmath::Vec3<int> indices, int texture_x, int texture_y, int texture_width, int texture_height, Material mat): 
                    indices(indices), shading(texture_x, texture_y, texture_width, texture_height), mat(mat), use_texture(true){}

    friend class Triangle;
};

class Trimesh: public Hitable {
private:
    TriInner* triangles;
    int n_triangles;
    VertexBuffer buffer;
    Texture texture;
public:
    Trimesh(std::vector<TriInner> triangles, VertexBuffer buffer, Texture texture);
    
    __device__
    Isect hit_local(const rmath::Ray<float>& local_ray) override;
};

class Triangle: public Hitable {
private:
    const TriInner& inner;
    Texture& texture;
    rmath::Vec3<float> vertices[3];
    rmath::Vec3<float> vert_norms[3];
public:
    __device__
    Triangle(const TriInner& inner, VertexBuffer buffer, Texture& texture);

    __device__
    Isect hit_local(const rmath::Ray<float>& local_ray) override;
};
}

#endif