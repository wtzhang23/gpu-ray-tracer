#include "procedural/perlin.h"
#include <random>
#include <cmath>
#include <functional>

namespace procedural {
CUDA_HOSTDEV
float interpolate(float a, float b, float w) {
    return w * a + (1 - w) * b;
}

CUDA_HOSTDEV
const rmath::Vec3<float>& Perlin::hash(int x, int y, int z) const {
    int hash_x = x % n_sample_vecs;
    assert(hash_x >= 0);
    int hash_xy = (permutation[hash_x] + y) % n_sample_vecs;
    assert(hash_xy >= 0);
    int hash_xyz = (permutation[hash_xy] + z) % n_sample_vecs;
    assert(hash_xyz >= 0);
    int hash = permutation[hash_xyz];
    assert(hash < n_sample_vecs);
    return sample_vecs[hash];
}

struct WeightGenerator {
    int ix;
    int iy;
    int iz;
    float mx;
    float my;
    float mz;
    const Perlin& p;

    CUDA_HOSTDEV
    static float smoothstep_remap(float delta) {
        return delta * delta * (3 - 2 * delta);
    }

    CUDA_HOSTDEV
    WeightGenerator(float x, float y, float z, const Perlin& p): ix((int) floor(x) % p.get_n_sample_vecs()),
                    iy((int) floor(y) % p.get_n_sample_vecs()), iz((int) floor(z) % p.get_n_sample_vecs()), p(p),
                    mx(smoothstep_remap(x - floor(x))), my(smoothstep_remap(y - floor(y))), mz(smoothstep_remap(z - floor(z))) {
        assert(abs(mx) < rmath::THRESHOLD + 1);
        assert(abs(my) < rmath::THRESHOLD + 1);
        assert(abs(mz) < rmath::THRESHOLD + 1);
    }

    CUDA_HOSTDEV
    float gen_weight(int dx, int dy, int dz) {
        float cx = ix + dx, cy = iy + dy, cz = iz + dz;
        rmath::Vec3<float> offset({dx - mx, dy - my, dz - mz});
        const rmath::Vec3<float>& wv = p.hash(cx, cy, cz);
        float w = rmath::dot(wv, offset.normalized());
        assert (w <= 1 + rmath::THRESHOLD && w >= -1 - rmath::THRESHOLD);
        return w;
    }
};

CUDA_HOSTDEV
float Perlin::sample(float x, float y, float z) const {
    // generate weights
    WeightGenerator g{x * n_sample_vecs / period, y * n_sample_vecs / period, z * n_sample_vecs / period, *this};
    float w000 = g.gen_weight(0, 0, 0);
    float w001 = g.gen_weight(0, 0, 1);
    float w010 = g.gen_weight(0, 1, 0);
    float w011 = g.gen_weight(0, 1, 1);
    float w100 = g.gen_weight(1, 0, 0);
    float w101 = g.gen_weight(1, 0, 1);
    float w110 = g.gen_weight(1, 1, 0);
    float w111 = g.gen_weight(1, 1, 1);

    // linearly interpolate dimensions
    float x00 = interpolate(w000, w100, g.mx);
    float x01 = interpolate(w001, w101, g.mx);
    float x10 = interpolate(w010, w110, g.mx);
    float x11 = interpolate(w011, w111, g.mx);
    float xy0 = interpolate(x00, x10, g.my);
    float xy1 = interpolate(x01, x11, g.my);
    float xyz = interpolate(xy0, xy1, g.mz);
    return amplitude * xyz;
}

Perlin::Perlin(int seed, int n_sample_vecs): amplitude(1.0f), period(1.0f), n_sample_vecs(n_sample_vecs) {
    std::mt19937 generator{seed};
    auto rng = std::bind(std::uniform_real_distribution<float>{}, generator);
    
    cudaMallocManaged(&sample_vecs, n_sample_vecs * sizeof(rmath::Vec3<float>));
    cudaMallocManaged(&permutation, n_sample_vecs * sizeof(int));
    for (int i = 0; i < n_sample_vecs; i++) {
        float theta = acos(2 * rng() - 1);
        float phi = 2 * rng() * M_PI;
        sample_vecs[i] = rmath::Vec3<float>({cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)}).normalized();
        permutation[i] = i;
    }

    auto rng_int = std::bind(std::uniform_int_distribution<unsigned>{}, generator);
    for (int i = 0; i < n_sample_vecs; i++) {
        int swap_idx = rng_int() % n_sample_vecs;
        int temp = permutation[i];
        permutation[i] = permutation[swap_idx];
        permutation[swap_idx] = temp;
    }
}

void Perlin::free(Perlin& p) {
    cudaFree(p.sample_vecs);
    cudaFree(p.permutation);
}
}