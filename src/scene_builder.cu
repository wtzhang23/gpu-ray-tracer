#include "scene_builder.h"
#include <thrust/scan.h>
#include "raymath/linear.h"
#include "iostream"
#include "rayenv/scene.h"
#include "rayenv/scene.cuh"
#include "rayprimitives/texture.cuh"
#include "rayprimitives/material.h"
#include "rayprimitives/material.cuh"
#include "rayprimitives/hitable.cuh"
#include "rayprimitives/trimesh.cuh"
#include "rayprimitives/light.cuh"
#include "gputils/alloc.h"
#include "assets.h"

namespace rtracer {

std::vector<rmath::Vec3<float>> generate_normals(const std::vector<rmath::Vec3<float>>& vertices,
                                                    const std::vector<rmath::Vec3<int>>& triangles) {
    std::vector<rmath::Vec3<float>> normals = std::vector<rmath::Vec3<float>>(vertices.size());
    for (const rmath::Vec3<int>& tri : triangles) {
        rmath::Vec3<float> a = vertices[tri[1]] - vertices[tri[0]];
        rmath::Vec3<float> b = vertices[tri[2]] - vertices[tri[0]];
        rmath::Vec3<float> n = rmath::cross(a, b).normalized();
        normals[tri[0]] += n;
        normals[tri[1]] += n;
        normals[tri[2]] += n;
    }

    // renormalize sums
    for (rmath::Vec3<float>& n : normals) {
        n = n.normalized();
    }
    return normals;
}

struct MeshConfig {
    rprimitives::Trimesh** meshes;
    int* ends;
    rmath::Vec3<int>* indices;
    rprimitives::Material* mats;
    rprimitives::Shade* shadings;
    rmath::Vec3<float>* mesh_pos;
    rmath::Quat<float>* mesh_rot;
    int n_meshes;
};

__global__
void build_meshes(MeshConfig* config) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < config->n_meshes; i += stride) {
        int begin = i == 0 ? 0 : config->ends[i - 1];
        int count = config->ends[i] - begin;
        rprimitives::TriInner* triangles = new rprimitives::TriInner[count];
        for (int j = 0; j < count; j++) {
            rprimitives::TriInner inner = rprimitives::TriInner(config->indices[begin + j], config->mats[begin + j], config->shadings[begin + j]);
            triangles[j] = inner;
        }
        rprimitives::Trimesh* mesh = new rprimitives::Trimesh(triangles, count);
        mesh->set_position(config->mesh_pos[i]);
        mesh->set_orientation(config->mesh_rot[i]);
        config->meshes[i] = mesh;
    } 
}

struct LightConfig {
    rprimitives::Light** lights;
    rmath::Vec3<float>* point_light_pos;
    rmath::Vec3<float>* dir_light_dir;
    rmath::Vec4<float>* point_light_col;
    rmath::Vec4<float>* dir_light_col;
    int n_points;
    int n_directional;
};

__global__
void build_lights(LightConfig* config) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < config->n_points + config->n_directional; i += stride) {
        rprimitives::Light* light;
        if (i > config->n_points) {
            rprimitives::PointLight* point_light = new rprimitives::PointLight();
            point_light->set_color(config->point_light_col[i]);
            point_light->set_pos(config->point_light_pos[i]);
            light = point_light;
        } else {
            int j = i - config->n_points;
            rprimitives::DirLight* dir_light = new rprimitives::DirLight();
            dir_light->set_color(config->dir_light_col[j]);
            dir_light->set_shine_dir(config->dir_light_dir[j]);
            light = dir_light;
        }
        config->lights[i] = light;
    }
}

renv::Scene* SceneBuilder::build_scene(renv::Canvas canvas, renv::Camera camera) {
    // load assets
    gputils::TextureBuffer4D<float> atlas = assets::read_png(atlas_path.c_str());

    // flatten meshes
    std::vector<rmath::Vec3<int>> flattened_triangles{};
    std::vector<rprimitives::Shade> flattened_shadings{};
    std::vector<rprimitives::Material> flattened_mats{};
    std::vector<rmath::Vec3<float>> flattened_mesh_pos{};
    std::vector<rmath::Quat<float>> flattened_mesh_rot{};
    std::vector<int> counts{};
    for (MeshBuilder& b : this->meshes) {
        assert(b.triangles.size() == b.shadings.size());
        assert(b.triangles.size() == b.mats.size());
        counts.push_back(b.triangles.size());
        for (int i = 0; i < b.triangles.size(); i++) {
            flattened_triangles.push_back(b.triangles[i]);
            flattened_shadings.push_back(b.shadings[i]);
            flattened_mats.push_back(b.mats[i]);
        }
        flattened_mesh_pos.push_back(b.pos);
        flattened_mesh_rot.push_back(b.rot);
    }

    // build vertex buffer
    std::vector<rmath::Vec3<float>> normals = generate_normals(vertices, flattened_triangles);
    rprimitives::VertexBuffer buffer{vertices, normals};

    // build meshes
    thrust::inclusive_scan(counts.data(), counts.data() + counts.size(), counts.data());
    int* ends = gputils::copy_to_gpu<int>(counts.data(), counts.size());
    rprimitives::Shade* dev_shadings = gputils::copy_to_gpu<rprimitives::Shade>(flattened_shadings.data(), 
                                                                                flattened_shadings.size());
    rprimitives::Material* dev_mats = gputils::copy_to_gpu<rprimitives::Material>(flattened_mats.data(),
                                                                                flattened_mats.size());
    rmath::Vec3<int>* dev_tris = gputils::copy_to_gpu<rmath::Vec3<int>>(flattened_triangles.data(), 
                                                                                flattened_triangles.size());
    rmath::Vec3<float>* dev_mesh_pos = gputils::copy_to_gpu<rmath::Vec3<float>>(flattened_mesh_pos.data(), 
                                                                                flattened_mesh_pos.size());
    rmath::Quat<float>* dev_mesh_rot = gputils::copy_to_gpu<rmath::Quat<float>>(flattened_mesh_rot.data(), 
                                                                                flattened_mesh_rot.size());
    rprimitives::Trimesh** hitables = (rprimitives::Trimesh**) gputils::create_buffer(counts.size(), sizeof(rprimitives::Trimesh*));
    int n_hitables = this->meshes.size();
    MeshConfig mesh_config = {
                hitables, 
                ends, 
                dev_tris, 
                dev_mats, 
                dev_shadings, 
                dev_mesh_pos,
                dev_mesh_rot,
                (int) counts.size(),
            };
    MeshConfig* mesh_config_ptr = gputils::copy_to_gpu(&mesh_config, 1);
    build_meshes<<<1, 512>>>(mesh_config_ptr);
    cudaFree(mesh_config_ptr);
    cudaFree(ends);
    cudaFree(dev_shadings);
    cudaFree(dev_mats);
    cudaFree(dev_tris);

    // create lights
    rmath::Vec3<float>* dev_point_light_pos = gputils::copy_to_gpu<rmath::Vec3<float>>(point_light_pos.data(), point_light_pos.size());
    rmath::Vec3<float>* dev_dir_light_dir = gputils::copy_to_gpu<rmath::Vec3<float>>(dir_light_dir.data(), dir_light_dir.size());
    rmath::Vec4<float>* dev_point_light_col = gputils::copy_to_gpu<rmath::Vec4<float>>(point_light_col.data(), point_light_col.size());
    rmath::Vec4<float>* dev_dir_light_col = gputils::copy_to_gpu<rmath::Vec4<float>>(dir_light_col.data(), dir_light_col.size());
    int n_point_lights = point_light_col.size();
    int n_dir_lights = dir_light_col.size();
    int n_lights = n_point_lights + n_dir_lights;
    rprimitives::Light** lights = (rprimitives::Light**) gputils::create_buffer(n_lights, sizeof(rprimitives::Light*));
    LightConfig light_config = {lights, dev_point_light_pos, dev_dir_light_dir, dev_point_light_col, dev_dir_light_col, n_point_lights, n_dir_lights};
    LightConfig* light_config_ptr = gputils::copy_to_gpu(&light_config, 1);
    build_lights<<<1, 1024>>>(light_config_ptr);
    cudaFree(dev_point_light_pos);
    cudaFree(dev_dir_light_dir);
    cudaFree(dev_point_light_col);
    cudaFree(dev_dir_light_col);
    cudaFree(light_config_ptr);

    // configure local scene
    renv::Scene local_scene = renv::Scene{canvas, camera, atlas, (rprimitives::Hitable**) hitables,
                                        n_hitables, lights, n_lights, buffer};
    local_scene.set_ambience(ambience);
    local_scene.set_recurse_depth(recurse_depth);
    renv::Scene* s = (renv::Scene*) gputils::create_buffer(1, sizeof(renv::Scene));
    cudaMemcpy(s, &local_scene, sizeof(renv::Scene), cudaMemcpyDefault);
    return s;
}

void SceneBuilder::build_cube(float scale, rmath::Vec3<float> pos, rmath::Quat<float> rot, 
                                            rprimitives::Shade shade, rprimitives::Material mat) {
    /*   e-----f
     *  /|    /|
     * a-----b |
     * | g---|-h
     * |/    |/
     * c-----d
     */
    const rmath::Vec3<float> _a{-0.5f, 0.5f, -0.5f};
    const rmath::Vec3<float> _b{0.5f, 0.5f, -0.5f};
    const rmath::Vec3<float> _c{-0.5f, -0.5f, -0.5f};
    const rmath::Vec3<float> _d{0.5f, -0.5f, -0.5f};
    const rmath::Vec3<float> _e{-0.5f, 0.5f, 0.5f};
    const rmath::Vec3<float> _f{0.5f, 0.5f, 0.5f};
    const rmath::Vec3<float> _g{-0.5f, -0.5f, 0.5f};
    const rmath::Vec3<float> _h{0.5f, -0.5f, 0.5f};
    rmath::Vec3<float> a = scale * _a;
    rmath::Vec3<float> b = scale * _b;
    rmath::Vec3<float> c = scale * _c;
    rmath::Vec3<float> d = scale * _d;
    rmath::Vec3<float> e = scale * _e;
    rmath::Vec3<float> f = scale * _f;
    rmath::Vec3<float> g = scale * _g;
    rmath::Vec3<float> h = scale * _h;

    MeshBuilder& builder = create_mesh(pos, rot);
    builder.add_triangle(rmath::Vec3<int>{add_vertex(a), add_vertex(b), add_vertex(d)},
                            shade, mat);
    builder.add_triangle(rmath::Vec3<int>{add_vertex(c), add_vertex(d), add_vertex(a)},
                            shade, mat);
    builder.add_triangle(rmath::Vec3<int>{add_vertex(a), add_vertex(b), add_vertex(e)},
                            shade, mat);
    builder.add_triangle(rmath::Vec3<int>{add_vertex(e), add_vertex(f), add_vertex(b)},
                            shade, mat);
    builder.add_triangle(rmath::Vec3<int>{add_vertex(d), add_vertex(h), add_vertex(b)},
                            shade, mat);
    builder.add_triangle(rmath::Vec3<int>{add_vertex(b), add_vertex(f), add_vertex(h)},
                            shade, mat);
    builder.add_triangle(rmath::Vec3<int>{add_vertex(c), add_vertex(g), add_vertex(a)},
                            shade, mat);
    builder.add_triangle(rmath::Vec3<int>{add_vertex(a), add_vertex(e), add_vertex(g)},
                            shade, mat);
    builder.add_triangle(rmath::Vec3<int>{add_vertex(g), add_vertex(h), add_vertex(e)},
                            shade, mat);
    builder.add_triangle(rmath::Vec3<int>{add_vertex(e), add_vertex(f), add_vertex(h)},
                            shade, mat);
}

}