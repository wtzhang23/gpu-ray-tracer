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
#include <initializer_list>
namespace rmath {
constexpr double THRESHOLD = 1E-6f;
template <typename T, int Dim>
class Vec;

template <typename T, int Dim>
CUDA_HOSTDEV
T dot(const Vec<T, Dim>& first, const Vec<T, Dim>& second);

template <typename T, int Dim>
class Vec {
private:
    T coords[Dim];
public:
    CUDA_HOSTDEV
    Vec() {
        for (int i = 0; i < Dim; i++) {
            coords[i] = 0;
        }
    }

    CUDA_HOSTDEV
    Vec(T coords[Dim]) {
        for (int i = 0; i < Dim; i++) {
            this->coords[i] = coords[i];
        }
    }

    CUDA_HOSTDEV
    Vec(std::initializer_list<T> dims) {
        int idx = 0;
        for (T c : dims) {
            if (idx >= Dim) {
                break;
            }
            coords[idx] = c;
            idx++;
        }
    }

    CUDA_HOSTDEV
    static Vec<T, Dim> zero() {
        return Vec<T, Dim>();
    }

    CUDA_HOSTDEV
    T& operator[](int idx) {
        return coords[idx];
    }

    CUDA_HOSTDEV
    T operator[](int idx) const {
        return coords[idx];
    }

    CUDA_HOSTDEV
    Vec<T, Dim>& operator+=(const Vec<T, Dim>& other) {
        for (int i = 0; i < Dim; i++) {
            coords[i] += other[i];
        }
        return *this;
    }

    CUDA_HOSTDEV
    Vec<T, Dim>& operator-=(const Vec<T, Dim>& other) {
        for (int i = 0; i < Dim; i++) {
            coords[i] -= other[i];
        }
        return *this;
    }

    CUDA_HOSTDEV
    Vec<T, Dim>& operator*=(const Vec<T, Dim>& other) {
        for (int i = 0; i < Dim; i++) {
            coords[i] *= other[i];
        }
        return *this;
    }

    CUDA_HOSTDEV
    Vec<T, Dim>& operator/=(const Vec<T, Dim>& other) {
        for (int i = 0; i < Dim; i++) {
            coords[i] /= other[i];
        }
        return *this;
    }

    CUDA_HOSTDEV
    Vec<T, Dim>& operator*=(T coef) {
        for (int i = 0; i < Dim; i++) {
            coords[i] *= coef;
        }
        return *this;
    }

    CUDA_HOSTDEV
    friend Vec<T, Dim> operator+(const Vec<T, Dim>& first, const Vec<T, Dim>& second) {
        Vec<T, Dim> first_cpy = first;
        first_cpy += second;
        return first_cpy;
    }

    CUDA_HOSTDEV
    friend Vec<T, Dim> operator-(const Vec<T, Dim>& first, const Vec<T, Dim>& second) {
        Vec<T, Dim> first_cpy = first;
        first_cpy -= second;
        return first_cpy;
    }

    CUDA_HOSTDEV
    friend Vec<T, Dim> operator*(const Vec<T, Dim>& first, const Vec<T, Dim>& second) {
        Vec<T, Dim> first_cpy = first;
        first_cpy *= second;
        return first_cpy;
    }

    CUDA_HOSTDEV
    friend Vec<T, Dim> operator/(const Vec<T, Dim>& first, const Vec<T, Dim>& second) {
        Vec<T, Dim> first_cpy = first;
        first_cpy /= second;
        return first_cpy;
    }

    CUDA_HOSTDEV
    friend Vec<T, Dim> operator*(T coef, const Vec<T, Dim>& vec) {
        Vec<T, Dim> vec_cpy = vec;
        vec_cpy *= coef;
        return vec_cpy;
    }

    CUDA_HOSTDEV
    friend Vec<T, Dim> operator-(const Vec<T, Dim>& vec) {
        return (T) -1 * vec;
    }

    CUDA_HOSTDEV
    T squared_norm() const {
        return rmath::dot<T, Dim>(*this, *this);
    }

    CUDA_HOSTDEV
    T len() const {
        return sqrt(squared_norm());
    }

    CUDA_HOSTDEV
    Vec<T, Dim> normalized() const {
        T len = this->len();
        if (len > THRESHOLD) {
            return (1 / len) * *this;
        } else {
            return Vec<T, Dim>();
        }
    }

    CUDA_HOSTDEV
    Vec<T, Dim> negated() const {
        return operator*=(-1);
    }

    friend std::ostream& operator<<(std::ostream& os, const Vec<T, Dim>& vec) {
        os << "[";
        for (int i = 0; i < Dim; i++) {
            os << vec[i];
            if (i < Dim - 1) {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }
};

template<typename T>
using Vec3 = Vec<T, 3>; 

template<typename T>
using Vec4 = Vec<T, 4>; 

template <typename T, int Dim>
CUDA_HOSTDEV
T dot(const Vec<T, Dim>& first, const Vec<T, Dim>& second) {
    T dot_product = 0;
    for (int i = 0; i < Dim; i++) {
        dot_product += first[i] * second[i];
    }
    return dot_product;
}

template<typename T>
CUDA_HOSTDEV
Vec<T, 3> cross(const Vec<T, 3>& first, const Vec<T, 3>& second) {
    float x = first[1] * second[2] - first[2] * second[1];
    float y = first[2] * second[0] - first[0] * second[2];
    float z = first[0] * second[1] - first[1] * second[0];
    Vec<T, 3> output = Vec<T, 3>({x, y, z});
    return output;
}

template<typename T>
CUDA_HOSTDEV
Vec<T, 3> reflect(const Vec<T, 3>& dir, const Vec<T, 3>& norm) {
    // TODO: test
    float d_len = dir.len();
    Vec<T, 3> d_norm = dir.normalized();
    Vec<T, 3> n_norm = norm.normalized();
    Vec<T, 3> projection = dot(d_norm, n_norm) * n_norm;
    Vec<T, 3> r_norm = (d_norm - 2 * n_norm).normalized();
    return d_len * r_norm;
}

template<typename T>
CUDA_HOSTDEV
Vec<T, 3> refract(const Vec<T, 3>& dir, const Vec<T, 3>& norm, T index_from, T index_to, bool& tir) {
    // TODO: test
    float d_len = dir.len();
    Vec<T, 3> d_norm = dir.normalized();
    Vec<T, 3> n_norm = norm.normalized();
    T index_ratio = index_from / index_to;
    T cosi = dot(d_norm, n_norm);
    T sint_2 = index_ratio * index_ratio * (1 - cosi * cosi);
    if (sint_2 > 1) {
        tir = true;
        return d_len * reflect(d_norm, n_norm);
    } else {
        tir = false;
        return d_len * (index_ratio * d_norm + (index_ratio * cosi - sqrt(1 - sint_2)) * n_norm);
    }
}

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
    T operator()(int row, int col) const {
        return inner[row][col];
    }

    CUDA_HOSTDEV
    Mat3<T>& operator*=(const Mat3<T>& other) {
        float inner[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0;
                for (int k = 0; k < 3; k++) {
                    sum += this->inner[i][k] * other.inner[k][j];
                }
                inner[i][j] = sum;
            }
        }
        this->inner = inner;
        return *this;
    } 

    CUDA_HOSTDEV
    friend Mat3<T> operator*(const Mat3<T>& mat1, const Mat3<T>& mat2) {
        Mat3<T> mat1_cpy = mat1;
        mat1_cpy *= mat2;
        return mat1_cpy;
    }

    CUDA_HOSTDEV
    friend Vec3<T> operator*(const Mat3<T>& arr, const Vec3<T>& to_multiply) {
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