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
class Trimesh;

template<typename T>
class Triangle {
private:
    rmath::Vec3<T> a;
    rmath::Vec3<T> b;
    rmath::Vec3<T> c;
    rmath::Vec3<T> n;

    CUDA_HOSTDEV
    Triangle(rmath::Vec3<T> vertex_a, rmath::Vec3<T> vertex_b, rmath::Vec3<T> vertex_c, rmath::Vec3<T> norm): a(vertex_a), b(vertex_b), c(vertex_c), n(norm){}
public:
    friend class Trimesh<T>;
};

template<typename T>
class Trimesh: private Entity<T> {
private:
    gputils::FlatVec<T, 3> vertices;
    gputils::FlatVec<T, 3> normals;
    gputils::FlatVec<int, 3> triangles;
public:
    CUDA_HOSTDEV
    Triangle<T> get_triangle(int i) {
        int triangle[3] = rmath::Vec3<T>(triangles.get_vec(i));
        rmath::Vec3<T> a = rmath::Vec3<T>(vertices.get_vec(triangle[0]));
        rmath::Vec3<T> b = rmath::Vec3<T>(vertices.get_vec(triangle[1]));
        rmath::Vec3<T> c = rmath::Vec3<T>(vertices.get_vec(triangle[2]));
        rmath::Vec3<T> n = rmath::Vec3<T>(normals.get_vec(i));
        return Triangle<T>(a, b, c, n);
    }
};

template <typename T>
void free_trimesh(Trimesh<T>& mesh) {
    gputils::free_vec(mesh.vertices);
    gputils::free_vec(mesh.normals);
    gputils::free_vec(mesh.triangles);
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
        for (int tri[3] : triangles) {
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

        gputils::FlatVec<T, 3> verts(nv);
        gputils::FlatVec<T, 3> norms(nv);
        gputils::FlatVec<int, 3> tris(nv);

        for (int i = 0; i < nv; i++) {
            rmath::Vec3<T> vertex = vertices[i];
            rmath::Vec3<T> norm = vert_norm[i];
            verts.set_vec(i, {vertex[0], vertex[1], vertex[2]});
            verts.set_vec(i, {norm[0], norm[1], norm[2]});
        }

        for (int i = 0; i < nt; i++) {
            int tri[3] = triangles[i];
            tris.set_vec(i, tri);
        }

        return Trimesh<T>{verts, norms, tris};
    }
};
}

#endif