#ifndef TRIMESH_H
#define TRIMESH_H

#include <vector>
#include <array>
#include "rayprimitives/entity.h"
#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayprimitives/material.h"
#include "rayprimitives/texture.h"
#include "gputils/alloc.h"
#include "gputils/flat_vec.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace rprimitives {
template<typename T>
class Trimesh: private Entity<T> {
private:
    gputils::TextureBuffer3D<T> vertices;
    gputils::TextureBuffer3D<T> normals;
    gputils::TextureBuffer3D<int> triangles;
public:
    CUDA_HOSTDEV
    const gputils::TextureBuffer3D<T>& get_vertices() const {
        return vertices;
    }

    CUDA_HOSTDEV
    const gputils::TextureBuffer3D<T>& get_normals() const {
        return normals;
    }

    CUDA_HOSTDEV
    const gputils::TextureBuffer3D<int>& get_triangles() const {
        return triangles;
    }
};

template <typename T>
void free_trimesh(Trimesh<T>& mesh) {
    gputils::free_texture_buffer(mesh.vertices);
    gputils::free_texture_buffer(mesh.normals);
    gputils::free_texture_buffer(mesh.triangles);
}

template <typename T>
class TrimeshBuilder {
private:
    std::vector<rmath::Vec3<T>> vertices;
    std::vector<std::array<int, 3>> triangles;
    Material<T> mat;
    Texture<T> texture;
public:
    TrimeshBuilder(): vertices(), triangles(), mat(), texture() {}

    int add_vertex(rmath::Vec3<T> v) {
        int idx = vertices.size();
        vertices.push_back(v);
        return idx;
    }

    int add_triangle(int t[3]) {
        int idx = triangles.size();
        vertices.push_back(t);
        return idx;
    }

    Trimesh<T> build() {
        const int nv = vertices.size();
        const int nt = triangles.size();
        std::vector<rmath::Vec3<T>> vert_norm = std::vector<rmath::Vec3<T>>(nv, rmath::Vec3<T>(0, 0, 0));
        for (std::array<int, 3> tri : triangles) {
            rmath::Vec3<T> v1 = vertices[tri[0]], v2 = vertices[tri[1]], v3 = vertices[tri[2]];
            rmath::Vec3<T> leg1 = v2 - v1;
            rmath::Vec3<T> leg2 = v3 - v1;
            rmath::Vec3<T> n = rmath::cross(leg1, leg2).normalized();
            vert_norm[tri[0]] += n;
            vert_norm[tri[1]] += n;
            vert_norm[tri[2]] += n;
        }

        for (int i = 0; i < nv; i++) {
            vert_norm[i] = vert_norm[i].normalized();
        }

        return Trimesh<T>{gputils::TextureBuffer3D<float>(vertices.data(), nv, 0), 
                          gputils::TextureBuffer3D<float>(vert_norm.data(), nv, 0),
                          gputils::TextureBuffer3D<int>(triangles.data(), nt, 0)};
    }
};
}

#endif