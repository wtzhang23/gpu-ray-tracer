#ifdef __CUDACC__
#include <math.h>
#include <cassert>
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

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
    Vec3(T coords[3]): coords(coords){}

    CUDA_HOSTDEV
    static Vec3 zero() {
        return Vec3(0, 0, 0);
    }

    CUDA_HOSTDEV
    T squared_norm() const {
        return Vec3::dot(*this, *this);
    }

    CUDA_HOSTDEV
    T len() const {
        return sqrt(squared_norm());
    }

    CUDA_HOSTDEV
    T normalize() {
        T len = len();
        if (len > THRESHOLD) {
            coords[0] /= len;
            coords[1] /= len;
            coords[2] /= len;
        }
    }

    CUDA_HOSTDEV
    T negate() {
        scale(-1);
    }

    CUDA_HOSTDEV
    T scale(T coef) {
        coords[0] *= coef;
        coords[1] *= coef;
        coords[2] *= coef;
    }

    CUDA_HOSTDEV
    T operator[](int idx) const {
        return coords[idx];
    }

    CUDA_HOSTDEV
    friend T dot(const Vec3<T>& first, const Vec3<T>& second) {
        return first.x * second.x + first.y * second.y + first.z * second.z;
    }

    CUDA_HOSTDEV
    friend Vec3<T> cross(const Vec3<T>& first, const Vec3<T>& second) {
        float x = first.y * second.z - first.z * second.y;
        float y = first.z * second.x - first.x * second.z;
        float z = first.x * second.y - first.y * second.x;
        Vec3 output = Vec3(x, y, z);
        assert(abs(Vec3<T>::dot(first, output)) <= THRESHOLD);
        assert(abs(Vec3<T>::dot(second, output)) <= THRESHOLD);
        return output;
    }
};

template <typename T>
class Mat3 {
private:
    T inner[3][3];
    CUDA_HOSTDEV
    T compute_cofactor(int row1, int row2, int col1, int col2) {
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
    Mat3(T inner[3][3]): inner(inner){}
    
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
        T minor2 = inner[0][1] * compute_cofactor(1, 2, 0, 3);
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
    friend Mat3<T> multiply(Mat3<T>& a, Mat3<T>& b) {
        float inner[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0;
                for (int k = 0; k < 3; k++) {
                    sum += a[i][k] * b[k][j];
                }
                inner[i][j] = sum;
            }
        }
        return Mat3<T>(inner);
    } 

    CUDA_HOSTDEV
    Vec3<T> multiplyVec3(Vec3<T>& to_multiply) {
        T coords[3];
        for (int i = 0; i < 3; i++) {
            T sum = 0;
            for (int j = 0; j < 3; j++) {
                sum += inner[i][j] * to_multiply[j];
            }
            coords[i] = sum;
        }
        return Vec3<T>(coords);
    }

    CUDA_HOSTDEV
    Mat3<T> inverse() const {
        Mat3<T> adj = adjugate();
        T det = determinant();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                adj.inner[i][j] /= det;
            }
        }
        return adj;
    }
};
}