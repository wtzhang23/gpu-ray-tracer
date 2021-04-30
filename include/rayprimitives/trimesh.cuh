#ifndef TRIMESH_CUH
#define TRIMESH_CUH

#include "rayprimitives/entity.h"
#include "raymath/linear.h"
#include "rayprimitives/material.h"
#include "rayprimitives/texture.h"
#include "rayprimitives/entity.h"
#include "rayprimitives/hitable.cuh"
#include "rayenv/scene.h"
#include "gputils/alloc.h"

namespace rprimitives {
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

    __device__
    rmath::Vec3<float> get_vertex(int i, VertexBuffer& buffer);

    __device__
    rmath::Vec3<float> get_normal(int i, VertexBuffer& buffer);

    __device__
    Isect tri_hit(const rmath::Ray<float>& ray, renv::Scene& scene);

    friend class Triangle;
    friend class TrimeshBuilder;
};

class Trimesh: public Hitable {
private:
    TriInner* triangles;
    int n_triangles;
public:
    __device__
    Trimesh(TriInner* triangles, int n_triangles): triangles(triangles),
                                    n_triangles(n_triangles){}
    
    __device__
    Isect hit_local(const rmath::Ray<float>& local_ray, renv::Scene& scene) override;

    __host__ __device__
    ~Trimesh() override {}
};

}

#endif