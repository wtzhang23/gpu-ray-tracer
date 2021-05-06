#ifndef PROCEDURAL_CUBE_WORLD_H
#define PROCEDURAL_CUBE_WORLD_H

#include <string>

namespace renv {
namespace gpu {

class Scene;

}
}

namespace procedural {
namespace gpu {
renv::gpu::Scene* generate(std::string config_path);
}
}

#endif