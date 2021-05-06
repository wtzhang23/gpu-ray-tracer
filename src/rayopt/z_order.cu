#include "rayopt/z_order.h"

namespace ropt {
CUDA_HOSTDEV
unsigned long z_order(rmath::Vec3<float> vec) {
    auto inv = -vec;
    unsigned int x = *((unsigned int*) &inv[0]), 
                  y = *((unsigned int*) &inv[1]), 
                  z = *((unsigned int*) &inv[2]); // negative to make positive numbers have a larger morton code than negative numbers
    unsigned int x_offset = 31, y_offset = 31, z_offset = 31;
    unsigned long t = 0;
    for (unsigned int i = 0; i < 64; i++) {
        t <<= 1;
        switch (i % 3) {
            case 0: {
                // handle x
                assert(x_offset >= 0);
                t |= (x >> x_offset) & 0b1;
                x_offset--;
                break;
            }
            case 1: {
                // handle y
                assert(y_offset >= 0);
                t |= (y >> y_offset) & 0b1;
                y_offset--;
                break;
            }
            case 2: {
                // handle z
                assert(z_offset >= 0);
                t |= (z >> z_offset) & 0b1;
                z_offset--;
                break;
            }
        }
    }
    return t;
}
}