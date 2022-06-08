cmake_minimum_required(VERSION 3.22)
project(graph_learning)

set(CMAKE_CXX_STANDARD 20)

if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    find_package(MKL REQUIRED)
    find_package(OpenMP REQUIRED)
    message(STATUS "Compile options: ${INTERFACE_COMPILE_OPTIONS}")
endif()

find_package(Armadillo REQUIRED NO_MODULE)
find_package(Matplot++ REQUIRED)

add_executable(graph_learning src/main.cpp src/graphs/graph.cpp src/graphs/graph.h)

if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(graph_learning PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(graph_learning PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(graph_learning $<LINK_ONLY:MKL::MKL>)
    target_link_libraries(graph_learning OpenMP::OpenMP_CXX)
endif()

target_link_libraries(graph_learning ${ARMADILLO_LIBRARIES} Matplot++::matplot)
