#ifndef CUBE_WORLD_H
#define CUBE_WORLD_H

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