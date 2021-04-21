#include <math.h>
#include <cassert>

namespace linear {
constexpr double THRESHOLD = 1E-6;

template <typename T>
class Vec3 {
private:
    T coords[3];
public:
    __host__ __device__
    Vec3(T x, T y, T z): coords{x, y, z} {}

    __host__ __device__
    Vec3(T coords[3]): coords(coords){}

    __host__ __device__
    static Vec3 zero() {
        return Vec3(0, 0, 0);
    }

    __host__ __device__
    T squared_norm() const {
        return Vec3::dot(*this, *this);
    }

    __host__ __device__
    T len() const {
        return sqrt(this.squared_norm);
    }

    __host__ __device__
    T normalize() {
        T len = len();
        if (len > THRESHOLD) {
            x /= len;
            y /= len;
            z /= len;
        }
    }

    __host__ __device__
    T negate() {
        this.x = -this.x;
        this.y = -this.y;
        this.z = -this.z;
    }

    __host__ __device__
    T scale(T coef) {
        this.x *= coef;
        this.y *= coef;
        this.z *= coef;
    }

    __host__ __device__
    T operator[](int idx) const {
        return coords[idx];
    }

    __host__ __device__
    friend static T dot(const Vec3<T>& first, const Vec3<T>& second) {
        return first.x * second.x + first.y * second.y + first.z * second.z;
    }

    __host__ __device__
    friend static Vec3<T> cross(const Vec3<T>& first, const Vec3<T>& second) {
        float x = first.y * second.z - first.z * second.y;
        float y = first.z * second.x - first.x * second.z;
        float z = first.x * second.y - first.y * second.x;
        Vec3 output = Vec3(x, y, z);
        assert(abs(Vec3<T>::dot(first, output)) <= THRESHOLD);
        assert(abs(Vec3<T>::dot(second, output)) <= THRESHOLD);
        return output;
    }
}

template <typename T>
class Mat3 {
private:
    T inner[3][3];
    __host__ __device__
    T compute_cofactor(int row1, int row2, int col1, int col2) {
        return inner[row1][col1] * inner[row2][col2] - inner[row1][col2] * inner[row2][col1];
    }

    __host__ __device__
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
    __host__ __device__
    Mat3(T inner[3][3]): inner(inner){}
    
    __host__ __device__
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

    __host__ __device__
    T determinant() const {
        T minor1 = inner[0][0] * compute_cofactor(1, 2, 1, 2);
        T minor2 = inner[0][1] * compute_cofactor(1, 2, 0, 3);
        T minor3 = inner[0][2] * compute_cofactor(1, 2, 0, 1);
        return minor1 - minor2 + minor3;
    }

    __host__ __device__
    Mat3<T> transpose() const {
        T new_inner[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                new_inner[i][j] = inner[j][i];
            }
        }
        return Mat3<T>(new_inner);
    }

    __host__ __device__
    friend static Mat3<T> multiply(Mat3<T>& a, Mat3<T>& b) {
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
    } 

    __host__ __device__
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

    __host__ __device__
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
}
}