#include "rayopt/cpu/bvh.h"
#include "rayopt/bounding_box.h"
#include "rayopt/z_order.h"
#include "rayenv/cpu/scene.h"
#include <algorithm>
#include <iostream>

namespace ropt {

namespace cpu {

BVH::BVH(std::vector<BoundingBox> org_boxes): boxes(org_boxes.size()), ordering() {
    std::vector<unsigned long> codes{};
    for (int i = 0; i < org_boxes.size(); i++) {
        ordering.push_back(i);
        unsigned long code;
        if (boxes[i].is_degenerate()) {
            code = ULONG_MAX;
        } else {
            rmath::Vec3<float> center = -boxes[i].center(); // negate since first bit indicates pos/neg
            code = ropt::z_order(center);
        }
        codes.push_back(code);
    }

    std::sort(ordering.begin(), ordering.end(), [&](int o1, int o2) {
        return codes[o1] < codes[o2];
    });

    // reorder boxes
    for (int i = 0; i < boxes.size(); i++) {
        boxes[i] = org_boxes[ordering[i]];
    }

    int level_idx = 0;
    int level_size = boxes.size();
    while (level_size >= 2) {
        for (int i = 0; i < (level_size >> 1); i++) {
            BoundingBox left = boxes[level_idx + 2 * i];
            BoundingBox right = boxes[level_idx + 2 * i + 1];
            boxes.push_back(ropt::merge(left, right));
        }
        level_idx += level_size;
        level_size >>= 1;
    }
}

void BVH::traverse_helper(const rmath::Ray<float>& r, std::function<void(int)> fn, int box_idx) const {
    int true_box_idx = boxes.size() - box_idx;
    const BoundingBox& cur_box = boxes[true_box_idx];
    float time;
    if (cur_box.intersects(r, time)) {
        int left_child = box_idx * 2;
        int right_child = box_idx * 2 + 1;
        if (left_child > boxes.size()) {
            fn(ordering[true_box_idx]);
        } else {
            assert(right_child <= boxes.size());
            traverse_helper(r, fn, left_child);
            traverse_helper(r, fn, right_child);
        }
    }
}

void BVH::traverse(const rmath::Ray<float>& r, std::function<void(int)> fn) const {
    traverse_helper(r, fn, 1);
}

}

}