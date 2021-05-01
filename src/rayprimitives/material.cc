#include "rayprimitives/material.h"

namespace rprimitives {
std::ostream& operator<<(std::ostream& os, const Material& mat) {
    os << "{Ke: " << mat.Ke 
       << ", Ka: " << mat.Ka 
       << ", Kd: " << mat.Kd 
       << ", Ks: " << mat.Ks 
       << ", Kt: " << mat.Kt 
       << ", Kr: " << mat.Kr 
       << ", alpha: " << mat.alpha
       << ", eta: " << mat.eta
       << "}";
    return os;
}
}