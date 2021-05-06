#ifndef PROCEDURAL_PERLIN_H
#define PROCEDURAL_PERLIN_H

#include <cmath>
#include "raymath/linear.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace procedural {

class Perlin {
private:
    float amplitude;
    float period;
    rmath::Vec3<float>* sample_vecs;
    int* permutation;
    int n_sample_vecs;
public:
    Perlin(int seed, int n_sample_vecs);

    static void free(Perlin& p);

    void set_amplitude(float a) {
        amplitude = a;
    }

    void set_period(float p) {
        period = p;
    }

    CUDA_HOSTDEV
    float sample(float x, float y, float z) const;

    CUDA_HOSTDEV
    int get_n_sample_vecs() const {
        return n_sample_vecs;
    }

    CUDA_HOSTDEV
    const rmath::Vec3<float>& hash(int x, int y, int z) const;
};
}

#endif