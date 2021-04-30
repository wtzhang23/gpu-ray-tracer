#include "rayprimitives/trimesh.cuh"
#include "rayprimitives/texture.cuh"
#include "rayprimitives/hitable.cuh"
#include "gputils/alloc.h"
#include <iostream>

namespace rprimitives {
template<bool point>
gputils::TextureBuffer4D<float> arr_vec_to_text(std::vector<rmath::Vec3<float>>& arr_vec) {
    std::vector<float> flattened = std::vector<float>();
    for (rmath::Vec3<float> vec : arr_vec) {
        flattened.push_back(vec[0]);
        flattened.push_back(vec[1]);
        flattened.push_back(vec[2]);
        flattened.push_back(point ? 1.0f : 0.0f);
    }
    float* buffer = flattened.data();
    return gputils::TextureBuffer4D<float>(buffer, arr_vec.size(), 0);

}

VertexBuffer::VertexBuffer(std::vector<rmath::Vec3<float>>& vertices, std::vector<rmath::Vec3<float>>& normals): 
        v(arr_vec_to_text<true>(vertices)), n(arr_vec_to_text<false>(normals)){}

__device__
Isect Trimesh::hit_local(const rmath::Ray<float>& local_ray) {
    Isect best_isect{};
    for (int i = 0; i < n_triangles; i++) {
        // TODO: use bvh tree
        Triangle tri = Triangle(triangles[i], buffer);
        Isect tri_isect = tri.tri_hit(local_ray);
        if (tri_isect.hit && (!best_isect.hit || best_isect.time > tri_isect.time)) {
            best_isect = tri_isect;
        }
    }
    return best_isect;
}

__device__
rmath::Vec3<float> Triangle::get_vertex(int i) {
    float4 v = tex1D<float4>(buffer.get_vertices().get_obj(), inner.indices[i]);
    return rmath::Vec3<float>({v.x, v.y, v.z});
}

__device__
rmath::Vec3<float> Triangle::get_normal(int i) {
    float4 n = tex1D<float4>(buffer.get_normals().get_obj(), inner.indices[i]);
    return rmath::Vec3<float>({n.x, n.y, n.z});
}

__device__
Isect Triangle::tri_hit(const rmath::Ray<float>& local_ray) {
    rmath::Vec3<float> a = get_vertex(0);
    rmath::Vec3<float> b = get_vertex(1);
    rmath::Vec3<float> c = get_vertex(2);
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
            rv.mat = inner.mat;
            rv.use_texture = inner.shading.use_texture;
            if (inner.shading.use_texture) {
                Shade::Data::TextData data = inner.shading.data.text_data;
                rv.shading.text_coords = rmath::Vec<float, 2>({data.texture_x + bary1 * data.texture_width,
                                                        data.texture_y + bary2 * data.texture_height});
                rv.norm = (bary0 * get_normal(0) + bary1 * get_normal(1) + bary2 * get_normal(2)).normalized();
            } else {
                rv.shading.color = inner.shading.data.col;
            }
        }
    }
    return rv;
}
}