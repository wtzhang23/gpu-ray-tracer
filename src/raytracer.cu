#include <thrust/scan.h>
#include "raymath/linear.h"
#include "raytracer.h"
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
static const char* ATLAS_PATH = "assets/sus.png";
static const int SQ_WIDTH = 16;

__global__
void trace(renv::Scene* scene) {
    renv::Canvas& canvas = scene->get_canvas();
    renv::Camera& cam = scene->get_camera();
    rprimitives::Texture& atlas = scene->get_atlas();
    
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int i = x; i < canvas.get_width(); i += stride_x) {
        for (int j = y; j < canvas.get_height(); j += stride_y) {
            rmath::Vec4<float> norm_col = get_color_from_texture(atlas, i, j);
            canvas.set_color(i, j, renv::Color(norm_col[0], norm_col[1], norm_col[2], norm_col[3]));
            rmath::Ray<float> r = cam.at(i, j);
            rprimitives::Isect isect{};
            renv::cast_ray(scene, r, isect);
            if (isect.hit) {
                rmath::Vec4<float> c = rprimitives::illuminate(r, isect, scene);
                canvas.set_color(i, j, renv::Color(c[0] > 1.0f ? 1.0f : c[0], 
                                                   c[1] > 1.0f ? 1.0f : c[1], 
                                                   c[2] > 1.0f ? 1.0f : c[2], 
                                                   c[3] > 1.0f ? 1.0f : c[3]));
            }
        }
    }
}

void update_scene(renv::Scene* scene) {
    dim3 dimBlock(SQ_WIDTH, SQ_WIDTH);
    int grid_dim_x = scene->get_canvas().get_width() / SQ_WIDTH;
    int grid_dim_y = scene->get_canvas().get_height() / SQ_WIDTH;
    dim3 dimGrid(grid_dim_x == 0 ? 1 : grid_dim_x, grid_dim_y == 0 ? 1 : grid_dim_y);
    trace<<<dimGrid, dimBlock>>>(scene);
    int rv = cudaDeviceSynchronize();
    assert(rv == 0);
    renv::Canvas& canvas = scene->get_canvas();
}

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

renv::Scene* build_scene(int width, int height) {
    renv::Canvas canvas{width, height};
    renv::Camera camera{M_PI / 4, 200.0f, canvas};

    // load assets
    gputils::TextureBuffer4D<float> atlas = assets::read_png(ATLAS_PATH);

    // build vertex buffer
    std::vector<rmath::Vec3<float>> vertices{};
    std::vector<rmath::Vec3<int>> triangles{};
    std::vector<rmath::Vec3<float>> normals;
    
    vertices.push_back(rmath::Vec3<float>({-.5f, -.5f, 0.0f}));
    vertices.push_back(rmath::Vec3<float>({.5f, -.5f, 0.0f}));
    vertices.push_back(rmath::Vec3<float>({-.5f, .5f, 0.0f}));
    triangles.push_back(rmath::Vec3<int>({0, 1, 2}));
    normals = generate_normals(vertices, triangles);

    rprimitives::VertexBuffer buffer{vertices, normals};

    // build meshes
    std::vector<rmath::Vec3<float>> init_trimesh_pos;
    std::vector<rmath::Quat<float>> init_trimesh_rot;
    std::vector<int> counts;
    std::vector<rprimitives::Shade> shadings;
    std::vector<rprimitives::Material> mats;
    init_trimesh_pos.push_back(rmath::Vec3<float>({0.0f, 0.0f, 5.0f}));
    init_trimesh_rot.push_back(rmath::Quat<float>::identity());
    counts.push_back(1);
    shadings.push_back(rprimitives::Shade(rmath::Vec4<float>{1.0f, 1.0f, 1.0f, 1.0f}));
    mats.push_back(rprimitives::Material(
        rmath::Vec4<float>(),
        rmath::Vec4<float>(),
        rmath::Vec4<float>({1.0f, 0.0f, 0.0f, 1.0f}),
        rmath::Vec4<float>({1.0f, 1.0f, 1.0f, 1.0f}),
        rmath::Vec4<float>(),
        rmath::Vec4<float>(),
        10.0f,
        0.0f
    ));
    thrust::inclusive_scan(counts.data(), counts.data() + counts.size(), counts.data()); // compute prefix sum for ends

    // copy mesh data to gpu
    int* ends = gputils::copy_to_gpu<int>(counts.data(), counts.size());
    rprimitives::Shade* dev_shadings = gputils::copy_to_gpu<rprimitives::Shade>(shadings.data(), shadings.size());
    rprimitives::Material* dev_mats = gputils::copy_to_gpu<rprimitives::Material>(mats.data(), mats.size());
    rmath::Vec3<int>* dev_tris = gputils::copy_to_gpu<rmath::Vec3<int>>(triangles.data(), triangles.size());
    rmath::Vec3<float>* dev_mesh_pos = gputils::copy_to_gpu<rmath::Vec3<float>>(init_trimesh_pos.data(), init_trimesh_pos.size());
    rmath::Quat<float>* dev_mesh_rot = gputils::copy_to_gpu<rmath::Quat<float>>(init_trimesh_rot.data(), init_trimesh_rot.size());

    rprimitives::Trimesh** meshes;
    cudaMallocManaged(&meshes, sizeof(rprimitives::Trimesh*) * counts.size());
    MeshConfig mesh_config = {
                meshes, 
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

    // free memory
    cudaFree(mesh_config_ptr);
    cudaFree(ends);
    cudaFree(dev_shadings);
    cudaFree(dev_mats);
    cudaFree(dev_tris);

    // copy meshes to hitables list
    rprimitives::Hitable** hitables;
    int n_hitables = counts.size();
    cudaMallocManaged(&hitables, sizeof(rprimitives::Hitable) * n_hitables);
    for (int i = 0; i < counts.size(); i++) {
        hitables[i] = meshes[i];
    }
    cudaFree(meshes);

    // create lights
    std::vector<rmath::Vec3<float>> point_light_pos{};
    std::vector<rmath::Vec3<float>> dir_light_dir{};
    std::vector<rmath::Vec4<float>> point_light_col{};
    std::vector<rmath::Vec4<float>> dir_light_col{};
    dir_light_dir.push_back(rmath::Vec3<float>({0.0f, -1.0f, 1.0f}));
    dir_light_col.push_back(rmath::Vec4<float>({1.0f, 1.0f, 1.0f, 1.0f}));

    // copy light data to gpu
    rmath::Vec3<float>* dev_point_light_pos = gputils::copy_to_gpu<rmath::Vec3<float>>(point_light_pos.data(), point_light_pos.size());
    rmath::Vec3<float>* dev_dir_light_dir = gputils::copy_to_gpu<rmath::Vec3<float>>(dir_light_dir.data(), dir_light_dir.size());
    rmath::Vec4<float>* dev_point_light_col = gputils::copy_to_gpu<rmath::Vec4<float>>(point_light_col.data(), point_light_col.size());
    rmath::Vec4<float>* dev_dir_light_col = gputils::copy_to_gpu<rmath::Vec4<float>>(dir_light_col.data(), dir_light_col.size());

    rprimitives::Light** lights;
    int n_point_lights = point_light_col.size();
    int n_dir_lights = dir_light_col.size();
    int n_lights = n_point_lights + n_dir_lights;
    cudaMallocManaged(&lights, sizeof(rprimitives::Light*) * n_lights);
    LightConfig light_config = {lights, dev_point_light_pos, dev_dir_light_dir, dev_point_light_col, dev_dir_light_col, n_point_lights, n_dir_lights};
    LightConfig* light_config_ptr = gputils::copy_to_gpu(&light_config, 1);
    build_lights<<<1, 1024>>>(light_config_ptr);

    // free memory
    cudaFree(dev_point_light_pos);
    cudaFree(dev_dir_light_dir);
    cudaFree(dev_point_light_col);
    cudaFree(dev_dir_light_col);
    cudaFree(light_config_ptr);

    // configure local scene
    renv::Scene local_scene = renv::Scene{canvas, camera, atlas, hitables, n_hitables, lights, n_lights, buffer};
    local_scene.set_ambience(rmath::Vec4<float>({1.0f, 1.0f, 1.0f, 1.0f}));

    renv::Scene* scene;
    cudaMallocManaged(&scene, sizeof(renv::Scene));
    cudaMemcpy(scene, &local_scene, sizeof(renv::Scene), cudaMemcpyDefault);
    return scene;
}
}