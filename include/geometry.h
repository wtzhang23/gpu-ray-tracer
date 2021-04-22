#ifndef GEOMETRY_H
#define GEOMETRY_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <cmath>
#include <cassert>
#include <iostream>
#include "linear.h"

extern const double THRESHOLD;

namespace geometry {
template <typename T>
class Quat {
private:
    linear::Vec4<T> inner;
public:
    CUDA_HOSTDEV
    Quat(linear::Vec4<T> inner): inner(inner) {}

    CUDA_HOSTDEV
    Quat(T r, T i, T j, T k): inner({r, i, j, k}) {}

    CUDA_HOSTDEV
    Quat(linear::Vec3<T> inner): Quat(0, inner[0], inner[1], inner[2]) {}
    
    CUDA_HOSTDEV
    Quat(linear::Vec3<T> axis, T theta) {
        T half_cos_theta = cos(theta / 2);
        T half_sin_theta = sin(theta / 2);
        T inner[4] = {half_cos_theta, axis[0] * half_sin_theta, axis[1] * half_sin_theta, axis[2] * half_sin_theta};
        this->inner = linear::Vec4<T>(inner);
    }

    static Quat<T> identity() {
        return Quat<T>(1, 0, 0, 0);
    }

    linear::Vec4<T> to_Vec4() const {
        return inner;
    }

    T& operator[](int idx) {
        return inner[idx];
    }

    friend T dot(Quat& a, Quat& b) {
        return linear::dot(a.inner, b.inner);
    }

    T r() const {
        return inner[0];
    }

    T i() const {
        return inner[1];
    }

    T j() const {
        return inner[2];
    }

    T k() const {
        return inner[3];
    }

    T len() const {
        return inner.len();
    }

    Quat<T> conjugate() const {
        return Quat<T>(r(), -i(), -j(), -k());
    }

    Quat<T> inverse() const {
        T sq_norm = inner.squared_norm();
        
        if (sq_norm < THRESHOLD) {
            return Quat<T>(0, 0, 0, 0);
        }

        T inv_sq_norm = 1 / sq_norm;
        return Quat<T>(r() * inv_sq_norm, i() * -inv_sq_norm, j() * -inv_sq_norm, k() * -inv_sq_norm);
    }

    Quat<T> normalized() const {
        T length = len();
        return Quat<T>(r() / length, i() / length, j() / length, k() / length);
    }

    CUDA_HOSTDEV
    Quat<T>& operator*=(const Quat<T>& other) {
        inner = linear::Vec4<T>({r() * other.r() - i() * other.i() - j() * other.j() - k() * other.k(),
                        i() * other.r() + r() * other.i() + j() * other.k() - k() * other.j(),
                        j() * other.r() + r() * other.j() + k() * other.i() - i() * other.k(),
                        k() * other.r() + r() * other.k() + i() * other.j() - j() * other.i()});
        return *this;
    }

    CUDA_HOSTDEV
    friend Quat<T> operator*(const Quat<T>& first, const Quat<T>& second) {
        Quat<T> first_cpy = first;
        first_cpy *= second;
        return first_cpy;
    }

    CUDA_HOSTDEV
    friend linear::Vec3<T> operator*(const Quat<T>& quat, const linear::Vec3<T>& vec) {
        T length = vec.len();
        Quat<T> prod = quat.normalized() * Quat<T>(vec) * quat.inverse();
        return length * linear::Vec3<T>({prod.i(), prod.j(), prod.k()}).normalized();
    }

    CUDA_HOSTDEV
    linear::Mat3<T> to_Mat3() const {
        T length = len();
        T ni = i() / length;
        T nj = j() / length;
        T nk = k() / length;
        T nr = r() / length;

        T ii = 2 * ni * ni, jj = 2 * nj * nj, kk = 2 * nk * nk, rr = 2 * nr * nr;
        T ri = 2 * nr * ni, rj = 2 * nr * nj, rk = 2 * nr * nk, ij = 2 * ni * nj, 
               ik = 2 * ni * nk, jk = 2 * nj * nk;
        T inner[3][3] = {{1 - (jj + kk), ij - rk, ik + rj},
                         {ij + rk, 1 - (ii + kk), jk - ri},
                         {ik - rj, jk + ri, 1 - (ii + jj)}};
        return linear::Mat3<T>(inner);
    }

    friend std::ostream& operator<<(std::ostream& os, const Quat<T>& q) {
        os << "[r: " << q.r() << ", i: " << q.i() << ", j: " << q.j() << ", k: " << q.k() << "]"; 
        return os;
    }
};
}

#endif