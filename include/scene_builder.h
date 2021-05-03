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
    int hitable_idx;
    std::vector<rmath::Vec3<int>> triangles;
    std::vector<rprimitives::Shade> shadings;
    std::vector<rprimitives::Material> mats;
    rmath::Vec3<float> pos;
    rmath::Quat<float> rot;
    MeshBuilder(int hi, rmath::Vec3<float> pos, rmath::Quat<float> rot): 
                        hitable_idx(hi), triangles(), shadings(), 
                        mats(), pos(pos), rot(rot) {}
    MeshBuilder(int hi): MeshBuilder(hi, rmath::Vec3<float>(), rmath::Quat<float>::identity()){}
public:
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
    std::vector<renv::Transformation> trans;
    std::string atlas_path;
public:
    SceneBuilder(std::string atlas_path): vertices(), meshes(), point_light_pos(),
                        dir_light_dir(), point_light_col(), dir_light_col(), 
                        trans(), atlas_path(atlas_path){}

    int add_vertex(rmath::Vec3<float> v) {
        int idx = vertices.size();
        vertices.push_back(v);
        return idx;
    }

    int add_vertex(float x, float y, float z) {
        return add_vertex(rmath::Vec3<float>({x, y, z}));
    }

    MeshBuilder& create_mesh(rmath::Vec3<float> pos, rmath::Quat<float> rot) {
        int hi = meshes.size();
        meshes.push_back(MeshBuilder{hi, pos, rot});
        return meshes.back();
    }

    MeshBuilder& create_mesh() {
        return create_mesh(rmath::Vec3<float>(), rmath::Quat<float>::identity());
    }

    renv::Transformation& add_trans(const MeshBuilder& builder) {
        trans.push_back(renv::Transformation{builder.hitable_idx});
        return trans.back();
    }

    MeshBuilder& build_cube(float scale, rprimitives::Shade shade, rprimitives::Material mat);

    void add_directional_light(rmath::Vec3<float> dir, rmath::Vec4<float> col) {
        dir_light_dir.push_back(dir);
        dir_light_col.push_back(col);
    }

    void add_point_light(rmath::Vec3<float> pos, rmath::Vec4<float> col) {
        point_light_pos.push_back(pos);
        point_light_col.push_back(col);
    }
    renv::Scene* build_scene(renv::Canvas canvas, renv::Camera camera);
};

}

#endif