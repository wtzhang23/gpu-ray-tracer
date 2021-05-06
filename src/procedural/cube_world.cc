#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include "procedural/cube_world.h"
#include "procedural/perlin.h"
#include "rayenv/gpu/scene.h"
#include "rayenv/canvas.h"
#include "raymath/geometry.h"
#include "raymath/linear.h"
#include "scene_builder.h"

namespace procedural {

const int DEFAULT_SEED = 42;
const int DEFAULT_GRID_SIZE = 8;
const int DEFAULT_WIDTH = 640;
const int DEFAULT_HEIGHT = 480;
const float DEFAULT_FOV = M_PI / 4;
const float DEFAULT_UNIT_LEN = 200;
const float DEFAULT_AMPLITUDE = 1.0f;

rmath::Vec3<float> read_vec3(const rapidjson::Value& v) {
    return rmath::Vec3<float>({v[0].GetFloat(), v[1].GetFloat(), v[2].GetFloat()});
}

rmath::Vec4<float> read_vec4(const rapidjson::Value& v) {
    auto& vec = v;
    return rmath::Vec4<float>({v[0].GetFloat(), v[1].GetFloat(), v[2].GetFloat(), v[3].GetFloat()});
}

struct CubeConfig {
    rtracer::SceneBuilder builder;
    renv::Camera cam;
    renv::Canvas canvas;
};

CubeConfig generate_config(rapidjson::Document& document) {
    int seed = DEFAULT_SEED;
    int grid_size = DEFAULT_GRID_SIZE;
    int width = DEFAULT_WIDTH, height = DEFAULT_HEIGHT;
    float fov = DEFAULT_FOV;
    float unit_length = DEFAULT_UNIT_LEN;
    if (document.HasMember("seed")) {
        seed = document["seed"].GetInt();
    }
    if (document.HasMember("grid_size")) {
        grid_size = document["grid_size"].GetInt();
    }
    if (document.HasMember("width")) {
        width = document["width"].GetInt();
    }
    if (document.HasMember("height")) {
        height = document["height"].GetInt();
    }
    if (document.HasMember("fov")) {
        fov = (float) (document["fov"].GetDouble() * M_PI) / 180;
    }
    if (document.HasMember("unit_length")) {
        unit_length = (float) document["unit_length"].GetDouble();
    }

    // build scene    
    std::string atlas{document["atlas"].GetString()};
    renv::Canvas canvas{width, height};
    renv::Camera cam{fov, unit_length, canvas};
    rtracer::SceneBuilder scene_builder{atlas};

    // build cubes
    int n_cubes = 0;
    if (document.HasMember("cubes")) {
        const rapidjson::Value& cubes_obj = document["cubes"];
        n_cubes = cubes_obj.Size();
        for (rapidjson::SizeType i = 0; i < n_cubes; i++) {
            const rapidjson::Value& cube = cubes_obj[i];
            rprimitives::Material mat{};
            rprimitives::TextureCoords txt_coords{};

            if (cube.HasMember("texture")) {
                // TODO: implement texture mapping
            }

            // load constants
            if (cube.HasMember("Ke")) {
                mat.set_Ke(1.0f / UINT8_MAX * read_vec4(cube["Ke"]));
            }
            if (cube.HasMember("Ka")) {
                mat.set_Ka(1.0f / UINT8_MAX * read_vec4(cube["Ka"]));
            }
            if (cube.HasMember("Kd")) {
                mat.set_Kd(1.0f / UINT8_MAX * read_vec4(cube["Kd"]));
            }
            if (cube.HasMember("Ks")) {
                mat.set_Ks(1.0f / UINT8_MAX * read_vec4(cube["Ks"]));
            }
            if (cube.HasMember("Kt")) {
                mat.set_Kt(read_vec4(cube["Kt"]));
            }
            if (cube.HasMember("Kr")) {
                mat.set_Kr(read_vec4(cube["Kr"]));
            }
            if (cube.HasMember("alpha")) {
                mat.set_alpha((float) cube["alpha"].GetDouble());
            }
            if (cube.HasMember("eta")) {
                mat.set_eta((float) cube["eta"].GetDouble());
            }

            scene_builder.build_cube(.999f,
                txt_coords,
                mat
            );
        }
    }

    // build lights
    if (document.HasMember("lights")) {
        const rapidjson::Value& lights = document["lights"];
        if (lights.HasMember("directional")) {
            const rapidjson::Value& directional = lights["directional"];
            int n_directional = directional.Size();
            for (int i = 0; i < n_directional; i++) {
                const rapidjson::Value& light = directional[i];
                scene_builder.add_directional_light(read_vec3(light["dir"]), 
                                    1.0f / UINT8_MAX * read_vec4(light["col"]));
            }
        }

        if (lights.HasMember("point")) {
            const rapidjson::Value& point = lights["point"];
            int n_point = point.Size();
            for (int i = 0; i < n_point; i++) {
                const rapidjson::Value& light = point[i];
                scene_builder.add_point_light(read_vec3(light["pos"]), 
                                1.0f / UINT8_MAX * read_vec4(light["col"]));
            }
        }
    }

    // build cube world
    float amplitude = DEFAULT_AMPLITUDE;
    if (document.HasMember("amplitude")) {
        amplitude = (float) document["amplitude"].GetDouble();
    }
    std::vector<float> last_heights{};
    for (int i = 0; i < grid_size * grid_size; i++) {
        last_heights.push_back(0.0f);
    }
    float max_height = 0.0f;
    for (int c = 0; c < n_cubes; c++) {
        procedural::Perlin perlin{seed, (grid_size + 4) / 5};
        perlin.set_amplitude(amplitude);
        perlin.set_period(grid_size);
        rtracer::MeshBuilder& mesh_builder = scene_builder.get_mesh_builder(c);
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                float x = i - grid_size / 2.0f;
                float z = j - grid_size / 2.0f;
                float y_off = floor(0.5f * (perlin.sample(i, j, 0.0f) + amplitude)) + 1;
                for (int d = 0; d < y_off; d++) {
                    float y = last_heights[i * grid_size + j] + d;
                    int tid = scene_builder.add_trans(mesh_builder);
                    scene_builder.get_transformation(tid).set_position({x, y, z});
                }
                last_heights[i * grid_size + j] += y_off;
                max_height = std::max(max_height, last_heights[i * grid_size + j]);
            }
        }
        procedural::Perlin::free(perlin);
    }

    cam.set_position({0.0f, max_height + 10.0f, -(float) grid_size / 2});
    cam.set_orientation(rmath::Quat<float>({1.0f, 0.0f, 0.0f}, 45));
    return {scene_builder, cam, canvas};
}

void finish_env(renv::Environment& env, rapidjson::Document& document) {
    if (document.HasMember("ambience")) {
        env.set_ambience(read_vec4(document["ambience"]));
    }
    if (document.HasMember("depth")) {
        env.set_recurse_depth(document["depth"].GetInt());
    }
    if (document.HasMember("distance_attenuation")) {
        const rapidjson::Value& atten = document["distance_attenuation"];
        float const_term = (float) atten["constant_term"].GetDouble();
        float linear_term = (float) atten["linear_term"].GetDouble();
        float quad_term = (float) atten["quadratic_term"].GetDouble();
        env.set_dist_atten(const_term, linear_term, quad_term);
    }
}

namespace gpu {

renv::gpu::Scene* generate(std::string config_path) {
    // read doc
    std::ifstream ifs(config_path.c_str());
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document document;
    document.ParseStream(isw);

    CubeConfig config = generate_config(document);
    renv::gpu::Scene* scene = config.builder.build_gpu_scene(config.canvas, config.cam);
    renv::Environment& env = scene->get_environment();
    finish_env(env, document);
    return scene;
}

}

}