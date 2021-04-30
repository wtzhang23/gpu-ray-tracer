#ifndef TRIMESH_CUH
#define TRIMESH_CUH

#include <vector>
#include <array>
#include <memory>
#include "rayprimitives/entity.h"
#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayprimitives/material.h"
#include "rayprimitives/texture.h"
#include "rayprimitives/entity.h"
#include "rayprimitives/hitable.cuh"
#include "gputils/alloc.h"
#include "gputils/flat_vec.h"

namespace rprimitives {
class VertexBuffer {
private:
    gputils::TextureBuffer4D<float> v;
    gputils::TextureBuffer4D<float> n;
public:
    VertexBuffer(std::vector<rmath::Vec3<float>>& vertices, std::vector<rmath::Vec3<float>>& normals);
    
    __host__ __device__
    gputils::TextureBuffer4D<float>& get_vertices() {
        return v;
    }

    __host__ __device__
    gputils::TextureBuffer4D<float>& get_normals() {
        return n;
    }
};

struct Shade {
    union Data {
        struct TextData {
            int texture_x;
            int texture_y;
            int texture_width;
            int texture_height;
        } text_data;
        rmath::Vec4<float> col;
        __host__ __device__
        Data(rmath::Vec4<float> col): col(col) {}
        __host__ __device__
        Data(int texture_x, int texture_y, int texture_width, int texture_height): text_data{texture_x, texture_y, texture_width, texture_height}{}
    } data;
    bool use_texture;
    __host__ __device__
    Shade(rmath::Vec4<float> col): data(col), use_texture(false) {}
    __host__ __device__
    Shade(int texture_x, int texture_y, int texture_width, int texture_height): data(texture_x, texture_y, texture_width, texture_height),
                            use_texture(true) {}
};

class TriInner {
private:
    rmath::Vec3<int> indices;
    Material mat;
    Shade shading;
public:
    __device__
    TriInner(): indices(), mat(), shading(rmath::Vec4<float>()){}

    __device__
    TriInner(rmath::Vec3<int> indices, Material mat, Shade shading): indices(indices), mat(mat), shading(shading){}
    __host__ __device__
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
public:
    __device__
    Trimesh(TriInner* triangles, int n_triangles, VertexBuffer buffer): triangles(triangles),
                                    n_triangles(n_triangles), buffer(buffer){}
    
    __device__
    Isect hit_local(const rmath::Ray<float>& local_ray) override;

    __host__ __device__
    ~Trimesh() override {}
};

class Triangle {
private:
    const TriInner& inner;
    VertexBuffer& buffer;
public:
    __device__
    Triangle(const TriInner& inner, VertexBuffer& buffer): inner(inner), buffer(buffer) {}

    __device__
    Isect tri_hit(const rmath::Ray<float>& ray);

    __device__
    rmath::Vec3<float> get_vertex(int i);

    __device__
    rmath::Vec3<float> get_normal(int i);
};
}

#endif