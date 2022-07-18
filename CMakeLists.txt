cmake_minimum_required(VERSION 3.22)
project(graph_learning)

set(CMAKE_CXX_STANDARD 20)

if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(MKL_THREADING tbb_thread)
    find_package(MKL REQUIRED)
    message(STATUS "Compile options: ${INTERFACE_COMPILE_OPTIONS}")
endif()

find_package(Armadillo REQUIRED NO_MODULE)
find_package(Matplot++ REQUIRED)

add_executable(graph_learning src/main.cpp src/graphs/graph.cpp src/graphs/graph.h extern/progressbar.h)

if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(graph_learning PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(graph_learning PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(graph_learning $<LINK_ONLY:MKL::MKL>)
endif()

target_link_libraries(graph_learning ${ARMADILLO_LIBRARIES} Matplot++::matplot)
