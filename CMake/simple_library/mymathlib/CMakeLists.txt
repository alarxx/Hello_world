cmake_minimum_required(VERSION 3.13)
project(mymathlib)

# Normal Libraries (STATIC and SHARED)
# but there are also Object, Interface, Imported and Alias Libraries
add_library(mymath STATIC adder.cpp)

# Install
set_target_properties(mymath PROPERTIES PUBLIC_HEADER adder.h)
install(TARGETS mymath ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/mymath
    )
