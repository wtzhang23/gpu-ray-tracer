#include "scene_builder.h"
#include "rayenv/cpu/scene.h"
#include "rayprimitives/cpu/texture.h"
#include "rayprimitives/cpu/trimesh.h"
#include "rayprimitives/cpu/light.h"
#include "rayprimitives/cpu/vertex_buffer.h"
#include "assets.h"

namespace rtracer {

std::vector<rmath::Vec3<float>> SceneBuilder::generate_normals() {
    std::vector<rmath::Vec3<float>> normals = std::vector<rmath::Vec3<float>>(vertices.size());
    for (const MeshBuilder& mesh : meshes) {
        for (const rmath::Vec3<int>& tri : mesh.triangles) {
            rmath::Vec3<float> a = vertices[tri[1]] - vertices[tri[0]];
            rmath::Vec3<float> b = vertices[tri[2]] - vertices[tri[0]];
            rmath::Vec3<float> n = rmath::cross(a, b).normalized();
            normals[tri[0]] += n;
            normals[tri[1]] += n;
            normals[tri[2]] += n;
        }
    }

    // renormalize sums
    for (rmath::Vec3<float>& n : normals) {
        n = n.normalized();
    }
    return normals;
}

renv::cpu::Scene* SceneBuilder::build_cpu_scene(renv::Canvas canvas, renv::Camera camera) {
    // load assets
    rprimitives::cpu::Texture atlas = assets::cpu::read_png(atlas_path.c_str());

    // create trimeshes
    std::vector<rprimitives::cpu::Hitable*> hitables{};
    for (MeshBuilder& b : this->meshes) {
        std::vector<rprimitives::cpu::TriInner> inners{};
        for (int i = 0; i < b.triangles.size(); i++) {
            rmath::Vec3<int> pts = b.triangles[i];
            rprimitives::TextureCoords coords = b.coords[i];
            rprimitives::Material mat = b.mats[i];
            rprimitives::cpu::TriInner inner{pts, mat, coords};
            inners.push_back(inner);
        }
        rprimitives::cpu::Trimesh* mesh = new rprimitives::cpu::Trimesh(inners);
        mesh->set_position(b.pos);
        mesh->set_orientation(b.rot);
        hitables.push_back((rprimitives::cpu::Hitable*) mesh);
    }
    std::vector<rmath::Vec3<float>> normals = generate_normals();
    rprimitives::cpu::VertexBuffer buffer{vertices, normals};

    // create lights
    std::vector<rprimitives::cpu::Light*> lights{};
    for (int i = 0; i < point_light_pos.size(); i++) {
        const rmath::Vec3<float>& pos = point_light_pos[i];
        const rmath::Vec4<float>& col = point_light_col[i];
        rprimitives::cpu::PointLight* l = new rprimitives::cpu::PointLight();
        l->set_pos(pos);
        l->set_color(col);
        lights.push_back(l);
    }
    for (int i = 0; i < dir_light_dir.size(); i++) {
        const rmath::Vec3<float>& dir = dir_light_dir[i];
        const rmath::Vec4<float>& col = dir_light_col[i];
        rprimitives::cpu::DirLight* l = new rprimitives::cpu::DirLight();
        l->set_shine_dir(dir);
        l->set_color(col);
        lights.push_back(l);
    }

    // copy transformations
    int n_trans = this->trans.size();
    renv::Transformation* trans = new renv::Transformation[n_trans];
    for (int i = 0; i < n_trans; i++) {
        trans[i] = this->trans[i];
    }

    // build environment
    renv::Environment env{canvas, camera, trans, n_trans};

    // create scene
    return new renv::cpu::Scene(env, atlas, hitables, lights, buffer);
}

}
