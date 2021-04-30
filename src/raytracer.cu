#include <thrust/scan.h>
#include "raymath/linear.h"
#include "raytracer.h"
#include "iostream"
#include "rayprimitives/texture.cuh"
#include "rayprimitives/material.h"
#include "rayprimitives/hitable.cuh"
#include "rayprimitives/trimesh.cuh"
#include "gputils/alloc.h"
#include "assets.h"

namespace rtracer {
static const char* ATLAS_PATH = "assets/sus.png";

__device__
rprimitives::Isect cast_ray(renv::Scene& scene, rmath::Ray<float> r) {
    // TODO: use bvh tree
    rprimitives::Isect best_hit{};
    rprimitives::Hitable** hitables = scene.get_hitables();
    for (int i = 0; i < scene.n_hitables(); i++) {
        rprimitives::Hitable* h = hitables[i];
        rprimitives::Isect cur_hit = h->hit(r, scene);
        if (cur_hit.hit && (!best_hit.hit || best_hit.time > cur_hit.time)) {
            best_hit = cur_hit;
        }
    }
    return best_hit;
}

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
            rprimitives::Isect isect = cast_ray(*scene, r);
            if (isect.hit) {
                canvas.set_color(i, j, renv::Color(1.0f, 1.0f, 1.0f, 1.0f));
            }
        }
    }
}

void update_scene(renv::Scene* scene) {
    dim3 dimBlock(32, 32);
    int grid_dim_x = scene->get_canvas().get_width() / 32;
    int grid_dim_y = scene->get_canvas().get_height() / 32;
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
    mats.push_back(rprimitives::Material{});
    thrust::inclusive_scan(counts.data(), counts.data() + counts.size(), counts.data()); // compute prefix sum for ends

    // copy data to gpu
    int* ends = gputils::copy_to_gpu<int>(counts.data(), counts.size());
    rprimitives::Shade* dev_shadings = gputils::copy_to_gpu<rprimitives::Shade>(shadings.data(), shadings.size());
    rprimitives::Material* dev_mats = gputils::copy_to_gpu<rprimitives::Material>(mats.data(), mats.size());
    rmath::Vec3<int>* dev_tris = gputils::copy_to_gpu<rmath::Vec3<int>>(triangles.data(), triangles.size());
    rmath::Vec3<float>* dev_mesh_pos = gputils::copy_to_gpu<rmath::Vec3<float>>(init_trimesh_pos.data(), init_trimesh_pos.size());
    rmath::Quat<float>* dev_mesh_rot = gputils::copy_to_gpu<rmath::Quat<float>>(init_trimesh_rot.data(), init_trimesh_rot.size());

    // create meshes
    rprimitives::Trimesh** meshes;
    cudaMallocManaged(&meshes, sizeof(rprimitives::Trimesh*) * counts.size());
    
    // create meshes
    MeshConfig config = {
                meshes, 
                ends, 
                dev_tris, 
                dev_mats, 
                dev_shadings, 
                dev_mesh_pos,
                dev_mesh_rot,
                (int) counts.size(),
            };
    MeshConfig* config_ptr;
    cudaMalloc(&config_ptr, sizeof(MeshConfig));
    cudaMemcpy(config_ptr, &config, sizeof(MeshConfig), cudaMemcpyHostToDevice);
    build_meshes<<<1, 512>>>(config_ptr);
    cudaDeviceSynchronize();

    cudaFree(config_ptr);
    cudaFree(ends);
    cudaFree(dev_shadings);
    cudaFree(dev_mats);
    cudaFree(dev_tris);

    std::vector<rprimitives::Hitable*> hitables = std::vector<rprimitives::Hitable*>{};
    for (int i = 0; i < counts.size(); i++) {
        hitables.push_back(meshes[i]);
    }
    cudaFree(meshes);

    renv::Scene local_scene = renv::Scene{canvas, camera, atlas, hitables, buffer};
    renv::Scene* scene;
    cudaMallocManaged(&scene, sizeof(renv::Scene));
    cudaMemcpy(scene, &local_scene, sizeof(renv::Scene), cudaMemcpyDefault);
    return scene;
}
}