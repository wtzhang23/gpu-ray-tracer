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

    rmath::Vec3<int> get_indices() const {
        return indices;
    }

    friend class Triangle;
    friend class TrimeshBuilder;
};

class Trimesh: public Hitable {
private:
    TriInner* triangles;
    int n_triangles;
    VertexBuffer buffer;

    static void free(Trimesh& mesh);
public:
    Trimesh(std::vector<TriInner> triangles, VertexBuffer buffer);
    
    __device__
    Isect hit_local(const rmath::Ray<float>& local_ray) override;
};

class Triangle: public Hitable {
private:
    const TriInner& inner;
    rmath::Vec3<float> vertices[3];
    rmath::Vec3<float> vert_norms[3];
public:
    __device__
    Triangle(const TriInner& inner, VertexBuffer buffer);

    __device__
    Isect hit_local(const rmath::Ray<float>& local_ray) override;
};

class TrimeshBuilder {
private:
    std::vector<rmath::Vec3<float>> vertices;
    std::vector<TriInner> triangles;
public:
    TrimeshBuilder(): vertices(), triangles() {}
    int add_vertex(rmath::Vec3<float> v);
    int add_triangle(TriInner t);
    std::vector<TriInner> build(std::vector<rmath::Vec3<float>>& tot_vert, std::vector<rmath::Vec3<float>>& tot_norm) const;
};
}

#endif