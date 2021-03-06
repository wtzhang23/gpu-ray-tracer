#ifndef RAYTRACER_H
#define RAYTRACER_H

namespace renv {
namespace gpu {

class Scene;

}

namespace cpu {

class Scene;

}
}

namespace rtracer {
namespace gpu {
    void update_scene(renv::gpu::Scene* scene, int kernel_dim, bool optimize);
    void debug_cast(renv::gpu::Scene* scene, int x, int y);
}

namespace cpu {
    void update_scene(renv::cpu::Scene* scene, int kernel_dim, bool optimize);
    void debug_cast(renv::cpu::Scene* scene, int x, int y);
}
}

#endif