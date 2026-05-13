cmake_minimum_required(VERSION 3.24)
project(nbody LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 120-real)   # RTX 5060 Ti = sm_120 Blackwell

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Vendored glad (web generator, v1 layout)
add_library(glad STATIC external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)
target_link_libraries(glad PUBLIC ${CMAKE_DL_LIBS})

add_executable(nbody
    src/main.cpp
    src/sim.cu
    src/renderer.cpp
)

# Copy shaders next to the binary so paths work from build dir
add_custom_command(TARGET nbody POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:nbody>/shaders
    COMMENT "Copying shaders to build directory"
)

target_include_directories(nbody PRIVATE src)
target_link_libraries(nbody PRIVATE
    OpenGL::GL glfw glad CUDA::cudart
)

target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo --use_fast_math>
)

set_target_properties(nbody PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA archs:   ${CMAKE_CUDA_ARCHITECTURES}")