#ifndef FLAT_VEC_H
#define FLAT_VEC_H

#include "gputils/alloc.h"
#include <array>
#include <initializer_list>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace gputils {
template<typename T, int Dim>
class FlatVec {
private:
    T* inner;
    int n_elems;
public:
    FlatVec(int n_elems): inner(gputils::create_buffer(Dim * n_elems, sizeof(T))), n_elems(n_elems){}

    CUDA_HOSTDEV
    int size() const {
        return n_elems;
    }

    CUDA_HOSTDEV
    std::array<T, Dim> get_vec(int idx) {
        std::array<T, Dim> vec;
        for (int i = 0; i < Dim; i++) {
            vec[i] = inner[i * n_elems + idx];
        }
        return vec;
    }

    CUDA_HOSTDEV
    void set_vec(int idx, T vec[Dim]) {
        for (int i = 0; i < Dim; i++) {
            inner[i * n_elems + idx] = vec[i];
        }
    }

    CUDA_HOSTDEV
    void set_vec(int idx, std::initializer_list<T> list) {
        int i = 0;
        for (T val : list) {
            inner[i * n_elems + idx] = val;
            i++;
        }
    }
};

template <typename T, int Dim>
void free_vec(FlatVec<T, Dim>& vec) {
    gputils::free_buffer((void*) vec.inner);
}
}

#endif