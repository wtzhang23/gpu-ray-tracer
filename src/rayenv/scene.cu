#include "rayenv/scene.h"
namespace renv {
Scene::Scene(Canvas& canvas, Camera camera, rprimitives::Texture atlas, std::vector<rprimitives::Hitable*> hitables, rprimitives::VertexBuffer buffer): 
                                            canvas(canvas), cam(camera), atlas(atlas), buffer(buffer), nh(hitables.size()) {
    if (!hitables.empty()) {
        cudaMallocManaged(&this->hitables, sizeof(rprimitives::Hitable*) * hitables.size());
        cudaMemcpy(this->hitables, hitables.data(), sizeof(rprimitives::Hitable*) * hitables.size(), cudaMemcpyHostToDevice);
    }
}
}