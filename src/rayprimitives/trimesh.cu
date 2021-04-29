#include "rayprimitives/trimesh.h"
#include "rayprimitives/texture.cuh"
#include "gputils/alloc.h"

namespace rprimitives {
template<bool point>
gputils::TextureBuffer4D<float> arr_vec_to_text(std::vector<rmath::Vec3<float>>& arr_vec) {
    std::vector<float> flattened = std::vector<float>(arr_vec.size());
    for (rmath::Vec3<float> vec : arr_vec) {
        flattened.push_back(vec[0]);
        flattened.push_back(vec[1]);
        flattened.push_back(vec[2]);
        flattened.push_back(point ? 1.0f : 0.0f);
    }
    return gputils::TextureBuffer4D<float>(flattened.data(), arr_vec.size(), 0);
}

VertexBuffer::VertexBuffer(std::vector<rmath::Vec3<float>>& vertices, std::vector<rmath::Vec3<float>>& normals): 
        v(arr_vec_to_text<true>(vertices)), n(arr_vec_to_text<false>(normals)){}

Trimesh::Trimesh(std::vector<TriInner> inners, VertexBuffer buffer): buffer(buffer) {
    cudaMallocManaged(&this->triangles, sizeof(TriInner) * inners.size());
    cudaMemcpy(this->triangles, inners.data(), sizeof(TriInner) * inners.size(), cudaMemcpyHostToDevice);
}

void Trimesh::free(Trimesh& mesh) {
    cudaFree(mesh.triangles);
}

__device__
Isect Trimesh::hit_local(const rmath::Ray<float>& local_ray) {
    Isect best_isect{};
    for (int i = 0; i < n_triangles; i++) {
        // TODO: use bvh tree
        Isect tri_isect = Triangle(triangles[i], buffer).hit(local_ray);
        if (tri_isect.hit && (!best_isect.hit || best_isect.time > tri_isect.time)) {
            best_isect = tri_isect;
        }
    }
    return best_isect;
}

__device__
Triangle::Triangle(const TriInner& inner, VertexBuffer buffer): 
                        Hitable(inner), inner(inner) {
    gputils::TextureBuffer4D<float> v_text = buffer.get_vertices();
    gputils::TextureBuffer4D<float> n_text = buffer.get_normals();
    float4 a = tex1D<float4>(v_text.get_obj(), inner.indices[0]);
    float4 b = tex1D<float4>(v_text.get_obj(), inner.indices[1]);
    float4 c = tex1D<float4>(v_text.get_obj(), inner.indices[2]);
    float4 an = tex1D<float4>(n_text.get_obj(), inner.indices[0]);
    float4 bn = tex1D<float4>(n_text.get_obj(), inner.indices[1]);
    float4 cn = tex1D<float4>(n_text.get_obj(), inner.indices[2]);
    this->vertices[0] = rmath::Vec3<float>({a.x, a.y, a.z}); 
    this->vertices[1] = rmath::Vec3<float>({b.x, b.y, b.z});
    this->vertices[2] = rmath::Vec3<float>({c.x, c.y, c.z});
    this->vert_norms[0] = rmath::Vec3<float>({an.x, an.y, an.z});
    this->vert_norms[1] = rmath::Vec3<float>({bn.x, bn.y, bn.z});
    this->vert_norms[2] = rmath::Vec3<float>({cn.x, cn.y, cn.z});
}

__device__
Isect Triangle::hit_local(const rmath::Ray<float>& local_ray) {
    rmath::Vec3<float> plane_norm = rmath::cross(vertices[0], vertices[1]);
    rmath::Plane<float> plane = rmath::Plane<float>(vertices[0], plane_norm);
    
    Isect rv{};
    if (plane.hit(local_ray, rv.time) && rv.time >= 0) {
        rmath::Vec3<float> isect_pt = local_ray.at(rv.time);
        float tri_area = rmath::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]).len();
        float bary0 = rmath::cross(vertices[2] - isect_pt, vertices[1] - isect_pt).len() / tri_area;
        float bary1 = rmath::cross(vertices[2] - isect_pt, vertices[0] - isect_pt).len() / tri_area;
        float bary2 = rmath::cross(vertices[0] - isect_pt, vertices[1] - isect_pt).len() / tri_area;
        if (abs(bary0 + bary1 + bary2 - 1.0f) <= rmath::THRESHOLD) {
            rv.hit = true;
            rv.mat = inner.mat;
            rv.use_texture = inner.use_texture;
            if (inner.use_texture) {
                TriInner::Shade::TextData data = inner.shading.text_data;
                rv.shading.text_coords = rmath::Vec<float, 2>({data.texture_x + bary1 * data.texture_width,
                                                        data.texture_y + bary2 * data.texture_height});
            } else {
                rv.shading.color = inner.shading.col;
            }
        }
    }
    return rv;
}

std::vector<TriInner> TrimeshBuilder::build(std::vector<rmath::Vec3<float>>& tot_vert, std::vector<rmath::Vec3<float>>& tot_norm) const {
    std::vector<rmath::Vec3<float>> loc_norm = std::vector<rmath::Vec3<float>>(vertices.size());
    std::vector<TriInner> triangles = this->triangles;
    for (TriInner tri : triangles) {
        rmath::Vec3<int> indices = tri.get_indices();
        rmath::Vec3<float> a = vertices[indices[0]];
        rmath::Vec3<float> b = vertices[indices[1]];
        rmath::Vec3<float> n = rmath::cross(a, b).normalized();
        loc_norm[indices[0]] += n;
        loc_norm[indices[1]] += n;
        loc_norm[indices[2]] += n;
    }

    int shift = tot_vert.size();

    // create triangles
    for (TriInner& tri : triangles) {
        tri.indices += rmath::Vec3<int>({shift, shift, shift});
    }

    // add vertices
    for (rmath::Vec3<float> n : loc_norm) {
        tot_norm.push_back(n.normalized());
    }

    for (rmath::Vec3<float> v : vertices) {
        tot_norm.push_back(v);
    }

    return triangles;
}
}