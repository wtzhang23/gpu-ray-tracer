#include <png.h>
#include <cstdio>
#include <csetjmp>
#include <cstdint>
#include <iostream>
#include "assets.h"
#include "rayprimitives/cpu/texture.h"

namespace assets {

png_bytep* read_png_raw(const char* filename, int& width, int& height) {
    FILE* fp = fopen(filename, "rb");

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info = png_create_info_struct(png);
    setjmp(png_jmpbuf(png));
    png_init_io(png, fp);
    png_read_info(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) {
        png_set_strip_16(png);
    }

    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    } else if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }

    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png);
    }

    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_filter(png, 0xFF, PNG_FILLER_AFTER);
    }

    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png);
    }

    png_read_update_info(png, info);

    png_bytep* img_raw = new png_bytep[height];
    for (int row = 0; row < height; row++) {
        img_raw[row] = new png_byte[png_get_rowbytes(png, info)];
    }

    png_read_image(png, img_raw);
    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);
    return img_raw;
}

namespace gpu {
gputils::TextureBuffer4D<float> read_png(const char* filename) {
    int width, height;
    png_bytep* img_raw = read_png_raw(filename, width, height);

    // load into texture buffer
    float* flat_norm_img = new float[width * height * 4];
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            png_byte* px = &img_raw[row][col * 4];
            float* flat_px = &flat_norm_img[(row * width + col) * 4];
            for (int c = 0; c < 4; c++) {
                flat_px[c] = (float) px[c] / UINT8_MAX;
            }
        }
        delete[] img_raw[row];
    }
    delete[] img_raw;
    gputils::TextureBuffer4D<float> rv = gputils::TextureBuffer4D<float>(flat_norm_img, width, height);
    delete[] flat_norm_img;
    return rv;
}

}

namespace cpu {
rprimitives::cpu::Texture read_png(const char* filename) {
    int width, height;
    png_bytep* img_raw = read_png_raw(filename, width, height);
    std::vector<rmath::Vec4<float>> rv{};

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            for (int col = 0; col < width; col++) {
                png_byte* px = &img_raw[row][col * 4];
                rmath::Vec4<float> color{};
                for (int c = 0; c < 4; c++) {
                    color[c] = (float) px[c] / UINT8_MAX;
                }
                rv.push_back(color);
            }
        }
    }

    return rprimitives::cpu::Texture(rv, width, height);
}

}

}

