cmake_minimum_required(VERSION 3.13)

project(application)

add_executable(app main.cpp)

# No need to provide a path, if you ‘installed’ the archive library in
# the standard location: /usr/local/lib and headers in /usr/local/include
# Check mymathlib/README.md
target_link_directories(app PRIVATE ${CMAKE_SOURCE_DIR}/../mymathlib/build/)

target_link_libraries(app PRIVATE mymath)
# mymath может найти только app, и видна она только для app

# PRIVATE vs. PUBLIC
# target_link_directories PRIVATE - поиск пути доступен только для app
# target_link_libraries PRIVATE - в бинарном файле библиотека будет доступна только для app, наследники не будут видеть определения


# Проблема указания полного пути в импорте #include <path>
# target_include_directories(<target> <visibility> <path>)?
# add_subdirectory(dir)?
# find_library()



