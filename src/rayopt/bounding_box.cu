#include "rayopt/bounding_box.h"
#include "rayprimitives/entity.h"

namespace ropt {
CUDA_HOSTDEV
void BoundingBox::fit_vertex(const rmath::Vec3<float>& v) {
    if (!nondegenerate) {
        min = v;
        max = v;
        nondegenerate = true;
        return;
    }
    
    for (int i = 0; i < 3; i++) {
        if (v[i] < min[i]) {
            min[i] = v[i];
        }
        if (v[i] > max[i]) {
            max[i] = v[i];
        }
    }
}

CUDA_HOSTDEV
BoundingBox& merge_into(BoundingBox& a, const BoundingBox& b) {
    if (!a.nondegenerate) {
        a = b;
        return a;
    } else if (!b.nondegenerate) {
        return a;
    }
    for (int i = 0; i < 3; i++) {
        if (a.min[i] > b.min[i]) {
            a.min[i] = b.min[i];
        }

        if (a.max[i] < b.max[i]) {
            a.max[i] = b.max[i];
        }
    }
    return a;
}

CUDA_HOSTDEV
BoundingBox merge(const BoundingBox& a, const BoundingBox& b) {
    BoundingBox a_cpy = a;
    merge_into(a_cpy, b);
    return a_cpy;
}

CUDA_HOSTDEV
BoundingBox from_local(const BoundingBox& a, const rprimitives::Entity& e) {
    BoundingBox rv{};
    if (!a.nondegenerate) {
        return rv;
    }
    rv.fit_vertex(e.point_from_local(a.min));
    rv.fit_vertex(e.point_from_local(a.max));
    return rv;
}

CUDA_HOSTDEV
bool BoundingBox::intersects(const rmath::Ray<float>& r, float& time) const {
    if (!nondegenerate) {
        return false;
    }

    // Kay/Kajiya algorithm
    const rmath::Vec3<float>& dir = r.direction();
    const rmath::Vec3<float>& origin = r.origin();
    float time_min = -INFINITY;
    float time_max = INFINITY;
    for (int axis = 0; axis < 3; axis++) {
        // ray parallel with plane
        if (dir[axis] == 0) {
            continue;
        }
        float time_near = (min[axis] - origin[axis]) / dir[axis];
        float time_far = (max[axis] - origin[axis]) / dir[axis];
        if (time_near > time_far) {
            // swap
            float temp = time_near;
            time_near = time_far;
            time_far = temp;
        }
        if (time_near > time_min) {
            time_min = time_near;
        }
        if (time_far < time_max) {
            time_max = time_far;
        }

        if (time_min > time_max || time_max < rmath::THRESHOLD) {
            return false;
        }
    }

    if (time_min >= 0) {
        time = time_min;
    } else {
        time = time_max;
    }
    return true;
}

std::ostream& operator<<(std::ostream& os, const BoundingBox& box) {
    os << "{min: " << box.min << ", max: " << box.max << "}";
    return os;
}
}