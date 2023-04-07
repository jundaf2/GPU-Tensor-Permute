cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(ENABLE_CUDA ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(DEFAULT_BUILD_TYPE "Release")

project(eigenDNN LANGUAGES C CXX CUDA)
set(${PROJECT_NAME}_VERSION 0.0.0)
enable_language(C CXX CUDA)

# Prohibit in-source build
if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(FATAL_ERROR " does not support in-source CMake builds at this time.")
endif (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})

find_package(CUDA REQUIRED)
find_package(CUDAToolkit)

set(CUDA_FLAGS "--expt-extended-lambda --expt-relaxed-constexpr --compiler-options -fPIC -g -G -Xcompiler -Wall")
set(CUDA_FLAGS "-gencode code=sm_86,arch=compute_86 ${CUDA_FLAGS}")
set(CUDA_FLAGS "-ccbin=${CMAKE_CXX_COMPILER} ${CUDA_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie -lpthread")

set(COMMON_FILES_DIR ${PROJECT_SOURCE_DIR}/common)
set(MAIN_FILE ${PROJECT_SOURCE_DIR}/main.cu ${PROJECT_SOURCE_DIR}/seqdata_permute.cuh)

# Declaring the files
set(ALL_SRC ${MAIN_FILE})
foreach(FNAME IN LISTS ALL_SRC)
    message(${FNAME})
endforeach()

add_compile_options(-Wno-unknown-pragmas)


set(CUDNN_INCLUDE_DIR "/usr/local/cuda/include")
set(CUDA_LIBRARY_DIR "/usr/local/cuda/lib64")
set(CUDNN_LIBRARY "/usr/local/cuda/lib64/libcudnn.so")

include(CTest)
add_executable(seqdata_permute ${ALL_SRC})
link_directories(${CUDA_LIBRARY_DIR})
include_directories(${CUDNN_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
target_include_directories(seqdata_permute PRIVATE SYSTEM ${HDR_FILES_DIR})
target_link_libraries(seqdata_permute PRIVATE ${CUDA_LIBRARIES} ${CUDNN_LIBRARY})
set_target_properties(seqdata_permute PROPERTIES POSITION_INDEPENDENT_CODE ON)
