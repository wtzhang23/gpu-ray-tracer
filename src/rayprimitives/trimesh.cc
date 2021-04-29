#include "rayprimitives/trimesh.h"

namespace rprimitives {
int TrimeshBuilder::add_vertex(rmath::Vec3<float> v) {
    int rv = vertices.size();
    vertices.push_back(v);
    return rv;
}

int TrimeshBuilder::add_triangle(TriInner t) {
    int rv = triangles.size();
    triangles.push_back(t);
    return rv;
}
}