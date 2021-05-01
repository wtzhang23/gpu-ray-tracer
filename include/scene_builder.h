#ifndef SCENE_BUILDER_H
#define SCENE_BUILDER_H

#include <vector>
#include "raymath/linear.h"
#include "raymath/geometry.h"
#include "rayprimitives/material.h"
#include "rayprimitives/texture.h"
#include "rayenv/scene.h"
#include "rayenv/canvas.h"

namespace rtracer {

class MeshBuilder {
private:
    std::vector<rmath::Vec3<int>> triangles;
    std::vector<rprimitives::Shade> shadings;
    std::vector<rprimitives::Material> mats;
    rmath::Vec3<float> pos;
    rmath::Quat<float> rot;
public:
    MeshBuilder(rmath::Vec3<float> pos, rmath::Quat<float> rot): triangles(), shadings(), 
                        mats(), pos(pos), rot(rot) {}
    void add_triangle(rmath::Vec3<int> tri, rprimitives::Shade shade, rprimitives::Material mat) {
        triangles.push_back(tri);
        shadings.push_back(shade);
        mats.push_back(mat);
    }

    friend class SceneBuilder;
};

class SceneBuilder {
private:
    std::vector<rmath::Vec3<float>> vertices;
    std::vector<MeshBuilder> meshes;
    std::vector<rmath::Vec3<float>> point_light_pos;
    std::vector<rmath::Vec3<float>> dir_light_dir;
    std::vector<rmath::Vec4<float>> point_light_col;
    std::vector<rmath::Vec4<float>> dir_light_col;
    std::string atlas_path;
    rmath::Vec4<float> ambience;
    int recurse_depth;
public:
    SceneBuilder(std::string atlas_path): vertices(), meshes(), point_light_pos(),
                        dir_light_dir(), point_light_col(), dir_light_col(), atlas_path(atlas_path), ambience(),
                        recurse_depth(0){}

    int add_vertex(rmath::Vec3<float> v) {
        int idx = vertices.size();
        vertices.push_back(v);
        return idx;
    }

    int add_vertex(float x, float y, float z) {
        return add_vertex(rmath::Vec3<float>({x, y, z}));
    }

    MeshBuilder& create_mesh(rmath::Vec3<float> pos, rmath::Quat<float> rot) {
        meshes.push_back(MeshBuilder{pos, rot});
        return meshes.back();
    }

    void build_cube(float scale, rmath::Vec3<float> pos, rmath::Quat<float> rot, 
                        rprimitives::Shade shade, rprimitives::Material mat);

    void add_directional_light(rmath::Vec3<float> dir, rmath::Vec4<float> col) {
        dir_light_dir.push_back(dir);
        dir_light_col.push_back(col);
    }

    void add_point_light(rmath::Vec3<float> pos, rmath::Vec4<float> col) {
        point_light_pos.push_back(pos);
        point_light_col.push_back(col);
    }

    void set_ambience(rmath::Vec4<float> ambience) {
        this->ambience = ambience;
    }

    void set_recurse_depth(float depth) {
        recurse_depth = depth;
    }

    renv::Scene* build_scene(renv::Canvas canvas, renv::Camera camera);
};

}

#endif