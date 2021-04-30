#include "rayprimitives/vertex_buffer.h"

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
}