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
#include <cmath>
#include "raymath/linear.h"

namespace rmath {
template <typename T>
class Quat {
private:
    Vec4<T> inner;

    T imaginary_len() const {
        return sqrt(i() * i() + j() * j() + k() * k());
    }
public:
    CUDA_HOSTDEV
    Quat(Vec4<T> inner): inner(inner) {}

    CUDA_HOSTDEV
    Quat(T i, T j, T k, T r): inner({i, j, k, r}) {}

    CUDA_HOSTDEV
    Quat(Vec3<T> inner): Quat(inner[0], inner[1], inner[2], 0) {}
    
    CUDA_HOSTDEV
    Quat(Vec3<T> axis, T theta) {
        T half_cos_theta = cos(0.5f * theta);
        T half_sin_theta = sin(0.5f * theta);
        T inner[4] = {axis[0] * half_sin_theta, axis[1] * half_sin_theta, axis[2] * half_sin_theta, half_cos_theta};
        this->inner = Vec4<T>(inner);
    }

    CUDA_HOSTDEV
    Quat(Mat3<T> rot_mat) {
        T t;
        if (rot_mat(2, 2) < 0) {
            if (rot_mat(0, 0) > rot_mat(1, 1)) {
                t = 1 + rot_mat(0, 0) - rot_mat(1, 1) - rot_mat(2, 2);
                inner = Vec4<T>({t, 
                            rot_mat(1, 0) + rot_mat(0, 1),
                            rot_mat(0, 2) + rot_mat(2, 0),
                            rot_mat(2, 1) - rot_mat(1, 2)});
            } else {
                t = 1 - rot_mat(0, 0) + rot_mat(1, 1) - rot_mat(2, 2);
                inner = Vec4<T>({rot_mat(1, 0) + rot_mat(0, 1),
                            t,
                            rot_mat(2, 1) + rot_mat(1, 2),
                            rot_mat(0, 2) - rot_mat(2, 0)});
            }
        } else {
            if (rot_mat(0, 0) < -rot_mat(1, 1)) {
                t = 1 - rot_mat(0, 0) - rot_mat(1, 1) + rot_mat(2, 2);
                inner = Vec4<T>({rot_mat(0, 2) + rot_mat(2, 0),
                            rot_mat(2, 1) + rot_mat(1, 2),
                            t,
                            rot_mat(1, 0) - rot_mat(0, 1)});
            } else {
                t = 1 + rot_mat(0, 0) + rot_mat(1, 1) + rot_mat(2, 2);
                inner = Vec4<T>({rot_mat(2, 1) - rot_mat(1, 2),
                            rot_mat(0, 2) - rot_mat(2, 0),
                            rot_mat(1, 0) - rot_mat(0, 1),
                            t});
            }
        }
        inner = (0.5 / sqrt(t) * inner).normalized();
    }

    CUDA_HOSTDEV
    static Quat<T> identity() {
        return Quat<T>(0, 0, 0, 1);
    }

    CUDA_HOSTDEV
    Vec4<T> to_Vec4() const {
        return inner;
    }

    CUDA_HOSTDEV
    T& operator[](int idx) {
        return inner[idx];
    }

    CUDA_HOSTDEV
    friend T dot(Quat& a, Quat& b) {
        return dot(a.inner, b.inner);
    }

    CUDA_HOSTDEV
    T r() const {
        return inner[3];
    }

    CUDA_HOSTDEV
    T i() const {
        return inner[0];
    }

    CUDA_HOSTDEV
    T j() const {
        return inner[1];
    }

    CUDA_HOSTDEV
    T k() const {
        return inner[2];
    }

    CUDA_HOSTDEV
    Vec3<T> axis() const {
        T im_len = imaginary_len();
        if (im_len < THRESHOLD) {
            return Vec3<T>({0, 0, 0});
        } else {
            return Vec3<T>({i() / im_len, j() / im_len, k() / im_len}).normalized();
        }
    }

    CUDA_HOSTDEV
    T angle() const {
        T im_len = imaginary_len();
        if (im_len < THRESHOLD || r() < THRESHOLD) {
            return 0;
        } else {
            return 2 * atan2(im_len, r());
        }
    }

    CUDA_HOSTDEV
    Quat<T> conjugate() const {
        return Quat<T>(-i(), -j(), -k(), r());
    }

    CUDA_HOSTDEV
    Quat<T> inverse() const {
        T sq_norm = inner.squared_norm();
        
        if (sq_norm < THRESHOLD) {
            return Quat<T>(0, 0, 0, 0);
        }

        T inv_sq_norm = 1 / sq_norm;
        return Quat<T>(i() * -inv_sq_norm, j() * -inv_sq_norm, k() * -inv_sq_norm, r() * inv_sq_norm);
    }

    CUDA_HOSTDEV
    Quat<T> normalized() const {
        return Quat<T>(inner.normalized());
    }

    CUDA_HOSTDEV
    Quat<T>& operator*=(const Quat<T>& other) {
        inner = Vec4<T>({i() * other.r() + r() * other.i() + j() * other.k() - k() * other.j(),
                        j() * other.r() + r() * other.j() + k() * other.i() - i() * other.k(),
                        k() * other.r() + r() * other.k() + i() * other.j() - j() * other.i(),
                        r() * other.r() - i() * other.i() - j() * other.j() - k() * other.k()});
        return *this;
    }

    CUDA_HOSTDEV
    friend Quat<T> operator*(const Quat<T>& first, const Quat<T>& second) {
        Quat<T> first_cpy = first;
        first_cpy *= second;
        return first_cpy;
    }

    CUDA_HOSTDEV
    friend Vec3<T> operator*(const Quat<T>& quat, const Vec3<T>& vec) {
        T length = vec.len();
        Quat<T> prod = quat.normalized() * Quat<T>(vec) * quat.inverse();
        return length * Vec3<T>({prod.i(), prod.j(), prod.k()}).normalized();
    }

    CUDA_HOSTDEV
    Mat3<T> to_Mat3() const {
        T length = inner.len();
        T ni = i() / length;
        T nj = j() / length;
        T nk = k() / length;
        T nr = r() / length;

        T ii = 2.0f * ni * ni, jj = 2.0f * nj * nj, kk = 2.0f * nk * nk;
        T ri = 2.0f * nr * ni, rj = 2.0f * nr * nj, rk = 2.0f * nr * nk, ij = 2.0f * ni * nj, 
               ik = 2.0f * ni * nk, jk = 2.0f * nj * nk;
        T inner[3][3] = {{1 - (jj + kk), ij - rk, ik + rj},
                         {ij + rk, 1 - (ii + kk), jk - ri},
                         {ik - rj, jk + ri, 1 - (ii + jj)}};
        return Mat3<T>(inner);
    }

    friend std::ostream& operator<<(std::ostream& os, const Quat<T>& q) {
        os << "[i: " << q.i() << ", j: " << q.j() << ", k: " << q.k() <<  ", r: " << q.r() << "]"; 
        return os;
    }
};

template <typename T>
class Ray {
private:
    Vec3<T> o;
    Vec3<T> d;
public:
    CUDA_HOSTDEV
    Ray(): o(), d() {}

    CUDA_HOSTDEV
    Ray(Vec3<T> origin, Vec3<T> direction): o(origin), d(direction.normalized()) {}

    CUDA_HOSTDEV
    Vec3<T> origin() const {
        return o;
    }

    CUDA_HOSTDEV
    Vec3<T> direction() const {
        return d;
    }

    CUDA_HOSTDEV
    Vec3<T> at(T dt) const {
        return o + dt * d;
    }
};

template <typename T>
class Plane {
private:
    Vec3<T> o;
    Vec3<T> n;
public:
    CUDA_HOSTDEV
    Plane(Vec3<T> origin, Vec3<T> norm): o(origin), n(norm.normalized()) {}

    CUDA_HOSTDEV
    Vec3<T> origin() const {
        return o;
    }

    CUDA_HOSTDEV
    Vec3<T> normal() const {
        return n;
    }

    CUDA_HOSTDEV
    bool hit(const Ray<T>& r, float& time) const {
        float denom = rmath::dot(r.direction(), n);
        if (abs(denom) < THRESHOLD) {
            return false;
        }
        time = (1.0f / denom) * rmath::dot(o - r.origin(), n);
        return true;
    }
};
}

#endif