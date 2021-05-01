#include "rayprimitives/trimesh.cuh"
#include "rayprimitives/texture.cuh"
#include "rayprimitives/hitable.cuh"
#include "rayprimitives/vertex_buffer.h"
#include "gputils/alloc.h"
#include <iostream>

namespace rprimitives {
__device__
bool Trimesh::hit_local(const rmath::Ray<float>& local_ray, renv::Scene* scene, Isect& isect) {
    bool hit = false;
    for (int i = 0; i < n_triangles; i++) {
        // TODO: use bvh tree
        hit |= triangles[i].tri_hit(local_ray, scene, isect);
    }
    return hit;
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
bool TriInner::tri_hit(const rmath::Ray<float>& local_ray, renv::Scene* scene, Isect& isect) {
    rmath::Vec3<float> a = get_vertex(0, scene->get_vertex_buffer());
    rmath::Vec3<float> b = get_vertex(1, scene->get_vertex_buffer());
    rmath::Vec3<float> c = get_vertex(2, scene->get_vertex_buffer());
    rmath::Vec3<float> plane_norm = rmath::cross(b - a, c - a);
    rmath::Plane<float> plane = rmath::Plane<float>(a, plane_norm);
    
    float time;
    if (plane.hit(local_ray, time) && time >= 0 && time < isect.time) {
        rmath::Vec3<float> isect_pt = local_ray.at(time);
        float tri_area = plane_norm.len();
        float bary0 = rmath::cross(c - isect_pt, b - isect_pt).len() / tri_area;
        float bary1 = rmath::cross(c - isect_pt, a - isect_pt).len() / tri_area;
        float bary2 = rmath::cross(a - isect_pt, b - isect_pt).len() / tri_area;
        if (abs(bary0 + bary1 + bary2 - 1.0f) <= rmath::THRESHOLD) {
            isect.mat = &mat;
            isect.shading = &shading;
            isect.uv = rmath::Vec<float, 2>({bary1, bary2});
            rmath::Vec3<float> n0 = get_normal(0, scene->get_vertex_buffer());
            rmath::Vec3<float> n1 = get_normal(1, scene->get_vertex_buffer());
            rmath::Vec3<float> n2 = get_normal(2, scene->get_vertex_buffer());
            isect.norm = (bary0 * n0 + bary1 * n1 + bary2 * n2).normalized();
            isect.time = time;
            return true;
        }
    }
    return false;
}

}