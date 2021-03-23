cmake_minimum_required(VERSION 3.17.0)
project(phaseretrieval)


set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SOURCES 
        main.cu
        "${SRC_DIR}/lodepng.cpp"
        "${SRC_DIR}/multilayer.cu"
        "${SRC_DIR}/kernels.cu"
        "${SRC_DIR}/blur.cu"
        "${SRC_DIR}/cudaDebug.cu"
        )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)

include_directories(${INCLUDES})

cuda_add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} 
    PUBLIC
        ${INCLUDES}
    PRIVATE
        ${Python3_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
        ${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        pthread
        Python3::Python)

set_property(TARGET ${PROJECT_NAME}
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)