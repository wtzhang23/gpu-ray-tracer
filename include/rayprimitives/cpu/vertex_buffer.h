#ifndef RAYPRIMITIVES_CPU_VERTEX_BUFFER_H
#define RAYPRIMITIVES_CPU_VERTEX_BUFFER_H

#include <vector>
#include "raymath/linear.h"

namespace rprimitives {
namespace cpu {

class VertexBuffer {
private:
    std::vector<rmath::Vec3<float>> v;
    std::vector<rmath::Vec3<float>> n;
public:
    VertexBuffer(std::vector<rmath::Vec3<float>>& vertices, 
                    std::vector<rmath::Vec3<float>>& normals): v(vertices), n(normals){}

    const rmath::Vec3<float>& get_vertex(int i) {
        return v[i];
    } 

    const rmath::Vec3<float>& get_normal(int i) {
        return n[i];
    }
};
}
}

#endif