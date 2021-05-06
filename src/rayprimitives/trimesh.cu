#include "rayenv/gpu/scene.h"
#include "rayenv/cpu/scene.h"
#include "rayprimitives/gpu/trimesh.cuh"
#include "rayprimitives/cpu/trimesh.h"
#include "rayprimitives/cpu/vertex_buffer.h"
#include <iostream>

namespace rprimitives {
namespace gpu {

__device__
bool Trimesh::hit_local(const rmath::Ray<float>& local_ray, renv::gpu::Scene* scene, Isect& isect) {
    bool hit = false;
    for (int i = 0; i < n_triangles; i++) {
        // TODO: use bvh tree
        hit |= triangles[i].tri_hit(local_ray, scene, isect);
    }
    return hit;
}

__device__
ropt::BoundingBox Trimesh::compute_bounding_box(renv::gpu::Scene* scene) {
    ropt::BoundingBox rv{};
    for (int i = 0; i < n_triangles; i++) {
        TriInner& tri = triangles[i];
        for (int j = 0; j < 3; j++) {
            rmath::Vec3<float> v = tri.get_vertex(j, scene->get_vertex_buffer());
            rv.fit_vertex(v);
        }
    }
    return ropt::from_local(rv, *this);
}

__device__
rmath::Vec3<float> TriInner::get_vertex(int i, VertexBuffer& buffer) {
    float4 v = tex1D<float4>(buffer.get_vertices().get_obj(), indices[i]);
    return rmath::Vec3<float>({v.x, v.y, v.z});
}

__device__
rmath::Vec3<float> TriInner::get_normal(int i, VertexBuffer& buffer) {
    float4 n = tex1D<float4>(buffer.get_normals().get_obj(), indices[i]);
    return rmath::Vec3<float>({n.x, n.y, n.z});
}

__device__
bool TriInner::tri_hit(const rmath::Ray<float>& local_ray, renv::gpu::Scene* scene, Isect& isect) {
    rmath::Vec3<float> a = get_vertex(0, scene->get_vertex_buffer());
    rmath::Vec3<float> b = get_vertex(1, scene->get_vertex_buffer());
    rmath::Vec3<float> c = get_vertex(2, scene->get_vertex_buffer());
    rmath::Triangle<float> tri = rmath::Triangle<float>(a, b, c);
    
    float time;
    rmath::Vec<float, 2> uv;
    if (tri.hit(local_ray, time, uv) && time >= rmath::THRESHOLD && time < isect.time) {
        isect.mat = &mat;
        isect.coords = &coords;
        isect.uv = uv;
        rmath::Vec3<float> n0 = get_normal(0, scene->get_vertex_buffer());
        rmath::Vec3<float> n1 = get_normal(1, scene->get_vertex_buffer());
        rmath::Vec3<float> n2 = get_normal(2, scene->get_vertex_buffer());
        float bary0 = 1.0f - uv[0] - uv[1];
        isect.norm = (bary0 * n0 + uv[0] * n1 + uv[1] * n2).normalized();
        isect.time = time;
        return true;
    }
    return false;
}

}

namespace cpu {

bool Trimesh::hit_local(const rmath::Ray<float>& local_ray, renv::cpu::Scene* scene, Isect& isect) const {
    bool hit = false;
    for (const TriInner& inner : triangles) {
        // TODO: use bvh tree
        hit |= inner.tri_hit(local_ray, scene, isect);
    }
    return hit;
}

ropt::BoundingBox Trimesh::compute_bounding_box(renv::cpu::Scene* scene) const {
    ropt::BoundingBox rv{};
    for (const TriInner& inner : triangles) {
        for (int j = 0; j < 3; j++) {
            rmath::Vec3<float> v = scene->get_vertex_buffer().get_vertex(inner.indices[j]);
            rv.fit_vertex(v);
        }
    }
    return ropt::from_local(rv, *this);
}

bool TriInner::tri_hit(const rmath::Ray<float>& local_ray, renv::cpu::Scene* scene, Isect& isect) const {
    VertexBuffer& buffer = scene->get_vertex_buffer();
    rmath::Vec3<float> a = buffer.get_vertex(indices[0]);
    rmath::Vec3<float> b = buffer.get_vertex(indices[1]);
    rmath::Vec3<float> c = buffer.get_vertex(indices[2]);
    rmath::Triangle<float> tri = rmath::Triangle<float>(a, b, c);
    
    float time;
    rmath::Vec<float, 2> uv;
    if (tri.hit(local_ray, time, uv) && time >= rmath::THRESHOLD && time < isect.time) {
        isect.mat = &mat;
        isect.coords = &coords;
        isect.uv = uv;
        rmath::Vec3<float> n0 = buffer.get_normal(indices[0]);
        rmath::Vec3<float> n1 = buffer.get_normal(indices[1]);
        rmath::Vec3<float> n2 = buffer.get_normal(indices[2]);
        float bary0 = 1.0f - uv[0] - uv[1];
        isect.norm = (bary0 * n0 + uv[0] * n1 + uv[1] * n2).normalized();
        isect.time = time;
        return true;
    }
    return false;
}

}
}