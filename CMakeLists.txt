include(FindPkgConfig)

cmake_minimum_required(VERSION 3.10)
configure_file(include/config.h.in include/config.h)
project(RayTracer VERSION 1.0 LANGUAGES CXX CUDA)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_subdirectory(src)
add_executable(RayTracer ${src})
target_compile_features(RayTracer PUBLIC cxx_std_11)
target_include_directories(RayTracer PUBLIC include PUBLIC lib)
set_target_properties(RayTracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(RayTracer PROPERTIES POSITION_INDEPENDENT_CODE ON)

# dependencies
find_package(PNG REQUIRED)
find_package(SDL2 REQUIRED)
pkg_search_module(SDL2TTF REQUIRED SDL2_ttf>=2.0.0)
find_package(RapidJSON REQUIRED)
target_link_libraries(RayTracer ${PNG_LIBRARIES} ${SDL2_LIBRARIES} ${RAPIDJSON_LIBRARIES} ${SDL2TTF_LIBRARIES})