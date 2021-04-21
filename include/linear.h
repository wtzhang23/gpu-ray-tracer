#ifndef LINEAR_H
#define LINEAR_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <cmath>
#include <cassert>
#include <iostream>

namespace linear {
constexpr double THRESHOLD = 1E-6;

template <typename T>
class Vec3 {
private:
    T coords[3];
public:
    CUDA_HOSTDEV
    Vec3(T x, T y, T z): coords{x, y, z} {}

    CUDA_HOSTDEV
    Vec3(T coords[3]): Vec3(coords[0], coords[1], coords[2]){}

    CUDA_HOSTDEV
    static Vec3 zero() {
        return Vec3(0, 0, 0);
    }

    CUDA_HOSTDEV
    static T dot(const Vec3<T>& first, const Vec3<T>& second) {
        return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
    }

    CUDA_HOSTDEV
    static Vec3<T> cross(const Vec3<T>& first, const Vec3<T>& second) {
        float x = first[1] * second[2] - first[2] * second[1];
        float y = first[2] * second[0] - first[0] * second[2];
        float z = first[0] * second[1] - first[1] * second[0];
        Vec3 output = Vec3(x, y, z);
        assert(abs(Vec3<T>::dot(first, output)) <= THRESHOLD);
        assert(abs(Vec3<T>::dot(second, output)) <= THRESHOLD);
        return output;
    }

    CUDA_HOSTDEV
    T squared_norm() const {
        return Vec3<T>::dot(*this, *this);
    }

    CUDA_HOSTDEV
    T len() const {
        return sqrt(squared_norm());
    }

    CUDA_HOSTDEV
    Vec3<T> normalized() const {
        T len = this->len();
        if (len > THRESHOLD) {
            return scaled(1 / len);
        } else {
            return Vec3<T>::zero();
        }
    }

    CUDA_HOSTDEV
    Vec3<T> negated() const {
        return scaled(-1);
    }

    CUDA_HOSTDEV
    Vec3<T> scaled(T coef) const {
        T new_coords[3];
        new_coords[0] = coords[0] * coef;
        new_coords[1] = coords[1] * coef;
        new_coords[2] = coords[2] * coef;
        return Vec3<T>(new_coords);
    }

    CUDA_HOSTDEV
    T operator[](int idx) const {
        return coords[idx];
    }

    friend std::ostream& operator<<(std::ostream& os, const Vec3<T>& vec) {
        os << "[" << vec[0] << ", " << vec[1] << ", " << vec[2] << "]"; 
        return os;
    }
};


template <typename T>
class Mat3 {
private:
    T inner[3][3];
    CUDA_HOSTDEV
    T compute_cofactor(int row1, int row2, int col1, int col2) const {
        return inner[row1][col1] * inner[row2][col2] - inner[row1][col2] * inner[row2][col1];
    }

    CUDA_HOSTDEV
    Mat3 adjugate() const {
        T new_inner[3][3];
        new_inner[0][0] = compute_cofactor(1, 2, 1, 2);
        new_inner[0][1] = -compute_cofactor(0, 2, 1, 2);
        new_inner[0][2] = compute_cofactor(0, 1, 1, 2);
        new_inner[1][0] = -compute_cofactor(1, 2, 0, 2);
        new_inner[1][1] = compute_cofactor(0, 2, 0, 2);
        new_inner[1][2] = -compute_cofactor(0, 1, 0, 2);
        new_inner[2][0] = compute_cofactor(1, 2, 0, 1);
        new_inner[2][1] = -compute_cofactor(0, 2, 0, 1);
        new_inner[2][2] = compute_cofactor(0, 1, 0, 1);
        return Mat3<T>(new_inner);
    }
public:
    CUDA_HOSTDEV
    Mat3(T inner[3][3]) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                this->inner[i][j] = inner[i][j];
            }
        }
    }
    
    CUDA_HOSTDEV
    static Mat3<T> identity() {
        T inner[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == j) {
                    inner[i][j] = 1;
                } else {
                    inner[i][j] = 0;
                }
            }
        }
        return Mat3<T>(inner);
    }

    CUDA_HOSTDEV
    T determinant() const {
        T minor1 = inner[0][0] * compute_cofactor(1, 2, 1, 2);
        T minor2 = inner[0][1] * compute_cofactor(1, 2, 0, 2);
        T minor3 = inner[0][2] * compute_cofactor(1, 2, 0, 1);
        return minor1 - minor2 + minor3;
    }

    CUDA_HOSTDEV
    Mat3<T> transpose() const {
        T new_inner[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                new_inner[i][j] = inner[j][i];
            }
        }
        return Mat3<T>(new_inner);
    }

    CUDA_HOSTDEV
    static Mat3<T> multiply(const Mat3<T>& a, const Mat3<T>& b) {
        float inner[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0;
                for (int k = 0; k < 3; k++) {
                    sum += a.inner[i][k] * b.inner[k][j];
                }
                inner[i][j] = sum;
            }
        }
        return Mat3<T>(inner);
    } 

    CUDA_HOSTDEV
    static Vec3<T> multiplyVec3(const Mat3<T>& arr, const Vec3<T>& to_multiply) {
        T new_coords[3];
        for (int i = 0; i < 3; i++) {
            T sum = 0;
            for (int j = 0; j < 3; j++) {
                sum += arr.inner[i][j] * to_multiply[j];
            }
            new_coords[i] = sum;
        }
        return Vec3<T>(new_coords);
    }

    CUDA_HOSTDEV
    Mat3<T> inverse() const {
        T det = determinant();
        if (det > THRESHOLD) {
            Mat3<T> adj = adjugate();
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    adj.inner[i][j] /= det;
                }
            }
            return adj;
        } else {
            T new_inner[3][3] = {{NAN, NAN, NAN}, {NAN, NAN, NAN}, {NAN, NAN, NAN}};
            return Mat3<T>(new_inner);
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Mat3<T>& vec) {
        os << "[" << vec.inner[0][0] << " " << vec.inner[0][1] << " " << vec.inner[0][2];
        for (int i = 1; i < 3; i++) {
            os << ";";
            for (int j = 0; j < 3; j++) {
                os << " " << vec.inner[i][j];
            }
        }
        os << "]";
        return os;
    }
};
}

#endif