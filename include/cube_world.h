#ifndef CUBE_WORLD_H
#define CUBE_WORLD_H

#include <string>
#include "rayenv/scene.h"

namespace cube_world {

renv::Scene* generate(std::string config_path);
    
}

#endif