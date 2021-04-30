#include "rayprimitives/trimesh.cuh"
#include "rayprimitives/texture.cuh"
#include "rayprimitives/hitable.cuh"
#include "rayprimitives/vertex_buffer.h"
#include "gputils/alloc.h"
#include <iostream>

namespace rprimitives {
__device__
Isect Trimesh::hit_local(const rmath::Ray<float>& local_ray, renv::Scene& scene) {
    Isect best_isect{};
    for (int i = 0; i < n_triangles; i++) {
        // TODO: use bvh tree
        Isect tri_isect = triangles[i].tri_hit(local_ray, scene);
        if (tri_isect.hit && (!best_isect.hit || best_isect.time > tri_isect.time)) {
            best_isect = tri_isect;
        }
    }
    return best_isect;
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
Isect TriInner::tri_hit(const rmath::Ray<float>& local_ray, renv::Scene& scene) {
    rmath::Vec3<float> a = get_vertex(0, scene.get_vertex_buffer());
    rmath::Vec3<float> b = get_vertex(1, scene.get_vertex_buffer());
    rmath::Vec3<float> c = get_vertex(2, scene.get_vertex_buffer());
    rmath::Vec3<float> plane_norm = rmath::cross(b - a, c - a);
    rmath::Plane<float> plane = rmath::Plane<float>(a, plane_norm);
    
    Isect rv{};
    if (plane.hit(local_ray, rv.time) && rv.time >= 0) {
        rmath::Vec3<float> isect_pt = local_ray.at(rv.time);
        float tri_area = plane_norm.len();
        float bary0 = rmath::cross(c - isect_pt, b - isect_pt).len() / tri_area;
        float bary1 = rmath::cross(c - isect_pt, a - isect_pt).len() / tri_area;
        float bary2 = rmath::cross(a - isect_pt, b - isect_pt).len() / tri_area;
        if (abs(bary0 + bary1 + bary2 - 1.0f) <= rmath::THRESHOLD) {
            rv.hit = true;
            rv.mat = mat;
            rv.use_texture = shading.use_texture;
            if (shading.use_texture) {
                Shade::Data::TextData data = shading.data.text_data;
                rv.shading.text_coords = rmath::Vec<float, 2>({data.texture_x + bary1 * data.texture_width,
                                                        data.texture_y + bary2 * data.texture_height});
                rmath::Vec3<float> n0 = get_normal(0, scene.get_vertex_buffer());
                rmath::Vec3<float> n1 = get_normal(1, scene.get_vertex_buffer());
                rmath::Vec3<float> n2 = get_normal(2, scene.get_vertex_buffer());
                rv.norm = (bary0 * n0 + bary1 * n1 + bary2 * n2).normalized();
            } else {
                rv.shading.color = shading.data.col;
            }
        }
    }
    return rv;
}

}