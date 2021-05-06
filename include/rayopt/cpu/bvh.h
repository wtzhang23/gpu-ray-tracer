#ifndef RAYOPT_CPU_BVH_H
#define RAYOPT_CPU_BVH_H

#include <vector>
#include <functional>
#include "rayopt/bounding_box.h"
#include "raymath/linear.h"
#include "raymath/geometry.h"

namespace renv {
namespace cpu {

class Scene;

}
}

namespace ropt {
namespace cpu {

class BVH {
private:
    std::vector<BoundingBox> boxes;
    std::vector<int> ordering;

    void traverse_helper(const rmath::Ray<float>& r, std::function<void(int)> fn, int box_idx) const;

public:
    BVH(): boxes(), ordering() {}
    BVH(std::vector<BoundingBox> org_boxes);

    bool empty() const {
        return boxes.empty();
    }

    void traverse(const rmath::Ray<float>& r, std::function<void(int)> fn) const;

    static void free(BVH& b);
    friend class BVHIterator;
};


}

}

#endif