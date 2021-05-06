#ifndef TRIMESH_CUH
#define TRIMESH_CUH

#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayprimitives/material.h"
#include "rayprimitives/entity.h"
#include "rayprimitives/gpu/hitable.cuh"
#include "rayprimitives/gpu/texture.h"
#include "rayprimitives/gpu/vertex_buffer.h"
#include "rayprimitives/texture_coords.h"
#include "gputils/alloc.h"

namespace renv {
namespace gpu {

class Scene;

}
}

namespace rprimitives {
namespace gpu {

class TriInner {
private:
    rmath::Vec3<int> indices;
    Material mat;
    TextureCoords coords;
public:
    __device__
    TriInner(): indices(), mat(), coords(){}

    __device__
    TriInner(rmath::Vec3<int> indices, Material mat, TextureCoords coords): indices(indices), mat(mat), coords(coords){}
    
    __host__ __device__
    rmath::Vec3<int> get_indices() const {
        return indices;
    }

    __device__
    rmath::Vec3<float> get_vertex(int i, VertexBuffer& buffer);

    __device__
    rmath::Vec3<float> get_normal(int i, VertexBuffer& buffer);

    __device__
    bool tri_hit(const rmath::Ray<float>& ray, renv::gpu::Scene* scene, Isect& isect);

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
    bool hit_local(const rmath::Ray<float>& local_ray, renv::gpu::Scene* scene, Isect& isect) override;

    __device__
    ropt::BoundingBox compute_bounding_box(renv::gpu::Scene* scene) override;

    __host__ __device__
    ~Trimesh() override {}
};

}
}

#endif