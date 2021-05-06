#ifndef RAYPRIMITIVES_MATERIAL_H
#define RAYPRIMITIVES_MATERIAL_H

#include "raymath/linear.h"
#include <iostream>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace rprimitives {
class Material {
private:
    rmath::Vec4<float> Ke;
    rmath::Vec4<float> Ka;
    rmath::Vec4<float> Kd;
    rmath::Vec4<float> Ks;    
    rmath::Vec4<float> Kt;
    rmath::Vec4<float> Kr;
    float alpha;
    float eta;
public:
    CUDA_HOSTDEV
    Material(rmath::Vec4<float> Ke, rmath::Vec4<float> Ka, rmath::Vec4<float> Kd, rmath::Vec4<float> Ks, rmath::Vec4<float> Kt, rmath::Vec4<float> Kr,
                float alpha, float eta): Ke(Ke), Ka(Ka), Kd(Kd), Ks(Ks), Kt(Kt), Kr(Kr), alpha(alpha), eta(eta){}
    
    CUDA_HOSTDEV
    Material(): Ke(rmath::Vec4<float>()), Ka(), Kd(), Ks(), Kt(), Kr(), alpha(0), eta(1){}

    void set_Ke(rmath::Vec4<float> Ke) {
        this->Ke = Ke;
    }

    void set_Ka(rmath::Vec4<float> Ka) {
        this->Ka = Ka;
    }

    void set_Kd(rmath::Vec4<float> Kd) {
        this->Kd = Kd;
    }

    void set_Ks(rmath::Vec4<float> Ks) {
        this->Ks = Ks;
    }

    void set_Kt(rmath::Vec4<float> Kt) {
        this->Kt = Kt;
    }

    void set_Kr(rmath::Vec4<float> Kr) {
        this->Kr = Kr;
    }

    void set_alpha(float alpha) {
        this->alpha = alpha;
    }

    void set_eta(float eta) {
        this->eta = eta;
    }

    CUDA_HOSTDEV
    const rmath::Vec4<float>& get_Ke() const {
        return Ke;
    }

    CUDA_HOSTDEV
    const rmath::Vec4<float>& get_Ka() const {
        return Ka;
    }

    CUDA_HOSTDEV
    const rmath::Vec4<float>& get_Kd() const {
        return Kd;
    }

    CUDA_HOSTDEV
    const rmath::Vec4<float>& get_Ks() const {
        return Ks;
    }

    CUDA_HOSTDEV
    const rmath::Vec4<float>& get_Kt() const {
        return Kt;
    }

    CUDA_HOSTDEV
    const rmath::Vec4<float>& get_Kr() const {
        return Kr;
    }

    CUDA_HOSTDEV
    float get_alpha() const {
        return alpha;
    }

    CUDA_HOSTDEV
    float get_eta() const {
        return eta;
    }

    CUDA_HOSTDEV
    bool reflective() const {
        return Kr[0] > 0.0f || Kr[1] > 0.0f || Kr[2] > 0.0f || Kr[3] > 0.0f;
    }

    CUDA_HOSTDEV
    bool refractive() const {
        return Kt[0] > 0.0f || Kt[1] > 0.0f || Kt[2] > 0.0f || Kt[3] > 0.0f;
    }

    friend std::ostream& operator<<(std::ostream& os, const Material& mat);
};
}

#endif