#ifndef CUBE_WORLD_H
#define CUBE_WORLD_H

#include <string>

namespace renv {
class Scene;
}

namespace procedural {

renv::Scene* generate(std::string config_path);
    
}

#endif