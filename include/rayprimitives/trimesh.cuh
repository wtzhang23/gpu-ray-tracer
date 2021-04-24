#ifndef TRIMESH_CUH
#define TRIMESH_CUH

#include "rayprimitives/trimesh.h"

namespace rprimitives {
template<typename T>
class Triangle {
private:
    rmath::Vec3<T> a;
    rmath::Vec3<T> b;
    rmath::Vec3<T> c;
    rmath::Vec3<T> na;
    rmath::Vec3<T> nb;
    rmath::Vec3<T> nc;
public:
    __device__
    Triangle(int idx, const Trimesh<T>& mesh) {
        int3 tri = tex2D<int3>(mesh.get_triangles().get_obj(), idx, 1); 
        const gputils::TextureBuffer3D<T>& vertices = mesh.get_vertices();
        const gputils::TextureBuffer3D<T>& normals = mesh.get_normals();
        float3 a = tex2D<float3>(vertices.get_obj(), tri.x, 1);
        float3 b = tex2D<float3>(vertices.get_obj(), tri.y, 1);
        float3 c = tex2D<float3>(vertices.get_obj(), tri.z, 1);
        float3 na = tex2D<float3>(normals.get_obj(), tri.x, 1);
        float3 nb = tex2D<float3>(normals.get_obj(), tri.y, 1);
        float3 nc = tex2D<float3>(normals.get_obj(), tri.z, 1);
        this->a = rmath::Vec3<float>(a.x, a.y, a.z);
        this->b = rmath::Vec3<float>(b.x, b.y, b.z);
        this->c = rmath::Vec3<float>(c.x, c.y, c.z);
        this->na = rmath::Vec3<float>(na.x, na.y, na.z);
        this->nb = rmath::Vec3<float>(nb.x, nb.y, nb.z);
        this->nc = rmath::Vec3<float>(nc.x, nc.y, nc.z);
    }
};

}

#endif