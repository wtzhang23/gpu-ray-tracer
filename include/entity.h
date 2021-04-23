#ifndef ENTITY_H
#define ENTITY_H

#include "raymath/linear.h"
#include "raymath/geometry.h"

namespace rentity {

template <typename T>
class Entity {
protected:
    rmath::Quat<T> o;
    rmath::Vec3<T> p;
public:
    Entity(rmath::Vec3<T> position, rmath::Quat<T> orientation): o(orientation), p(position) {}

    void translate_global(rmath::Vec3<T> dp) {
        p += dp;
    }

    void translate(rmath::Vec3<T> dp) {
        translate_global(o.to_Mat3() * dp);
    }

    void rotate(rmath::Quat<T> dr) {
        o = dr * o;
    }

    void set_position(rmath::Vec3<T> p) {
        this->p = p;
    }

    void set_orientation(rmath::Quat<T> o) {
        this->o = o;
    }
};

}

#endif