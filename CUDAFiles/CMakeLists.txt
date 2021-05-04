cmake_minimum_required(VERSION 3.16.0)
project(phaseretrieval)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SOURCES 
        main.cpp
        "${SRC_DIR}/lodepng.cpp"
        "${SRC_DIR}/appData.cpp"
        "${SRC_DIR}/fista.cu"
        "${SRC_DIR}/kernels.cu"
        "${SRC_DIR}/blur.cu"
        "${SRC_DIR}/cudaDebug.cu"
        "${SRC_DIR}/videoParser.cpp"
        "${SRC_DIR}/utils.cpp"
        "${SRC_DIR}/settings.cpp"
        "${SRC_DIR}/ticker.cpp"
        )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -arch compute_30 -code sm_30)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui)

include_directories(${INCLUDES} ${OpenCV_INCLUDE_DIRS})

cuda_add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} 
    PUBLIC
        ${INCLUDES}
    )

target_link_libraries(${PROJECT_NAME}
        ${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        pthread
        ${OpenCV_LIBS})

set_property(TARGET ${PROJECT_NAME}
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)