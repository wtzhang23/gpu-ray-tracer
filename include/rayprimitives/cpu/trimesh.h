#ifndef RAYPRIMITIVES_CPU_TRIMESH_H
#define RAYPRIMITIVES_CPU_TRIMESH_H

#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayprimitives/material.h"
#include "rayprimitives/entity.h"
#include "rayprimitives/cpu/hitable.h"
#include "rayprimitives/cpu/texture.h"
#include "rayprimitives/cpu/vertex_buffer.h"
#include "rayprimitives/texture_coords.h"

namespace renv {
namespace cpu {

class Scene;

}
}

namespace rprimitives {
namespace cpu {

class TriInner {
private:
    rmath::Vec3<int> indices;
    Material mat;
    TextureCoords coords;
public:
    TriInner(): indices(), mat(), coords(){}
    TriInner(rmath::Vec3<int> indices, Material mat, TextureCoords coords): 
                                    indices(indices), mat(mat), coords(coords){}
    rmath::Vec3<int> get_indices() const {
        return indices;
    }
    rmath::Vec3<float> get_vertex(int i, VertexBuffer& buffer);
    rmath::Vec3<float> get_normal(int i, VertexBuffer& buffer);
    bool tri_hit(const rmath::Ray<float>& ray, renv::cpu::Scene* scene, Isect& isect) const;

    friend class Trimesh;
    friend class TrimeshBuilder;
};

class Trimesh: public Hitable {
private:
    std::vector<TriInner> triangles;
public:
    Trimesh(std::vector<TriInner> triangles): triangles(triangles){}
    bool hit_local(const rmath::Ray<float>& local_ray, renv::cpu::Scene* scene, Isect& isect) const override;
    ropt::BoundingBox compute_bounding_box(renv::cpu::Scene* scene) const override;
};

}
}

#endif