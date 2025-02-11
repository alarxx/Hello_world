message("Root CMakeLists.txt")

# generators
# cmake --install . instead use make install
cmake_minimum_required(VERSION 3.15)

# set the project name, version and languages
# By default, LANGUAGES C CXX, keeping only CXX
project(Tutorial VERSION 1.0 LANGUAGES CXX)

# --- Specify the Version of C++ Standard ---

# set(CMAKE_CXX_STANDARD 98)
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)
# INTERFACE library для распространения своих свойств
add_library(tutorial_compiler_flags INTERFACE)
# flag: -std=c++98
# Но у меня остается C++17:
# target_compile_features(tutorial_compiler_flags INTERFACE cxx_std_98)
# this works:
target_compile_options(tutorial_compiler_flags INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-std=c++11>")

# ------

# --- Generator expressions ---

# $<COMPILE_LANG_AND_ID:language,compiler_ids>
set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,ARMClang,LCC>")
# Microsoft Visual C++, другие флаги
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
# add Warning flags to `INTERFACE library`
target_compile_options(tutorial_compiler_flags INTERFACE
    "$<${gcc_like_cxx}:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>"
    "$<${msvc_cxx}:-W3>"
)
# Almost the same add Warning flags, but only when building, and not for installed versions
# target_compile_options(tutorial_compiler_flags INTERFACE
#     "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>"
#     "$<${msvc_cxx}:$<BUILD_INTERFACE:-W3>>"
# )

# Although, вместо генераторов можно использовать обычные if() ... endif(),
# see: https://stackoverflow.com/questions/10046114/in-cmake-how-can-i-test-if-the-compiler-is-clang

# C++
message(STATUS "Compiler used: ${CMAKE_CXX_COMPILER}")
# IDK why it doesn't work
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")
# Check compiler: GNU, Clang, AppleClang, MSVC, Intel
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID}")
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using clang with clang-cl ("MVSC") or clang native ("GNU")
    message(STATUS "Compiler Frontend Variant: ${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}")
endif()

# -------

# Single source of truth
set(MY_CUSTOM_VARIABLE "MY CUSTOM VARIABLE")
configure_file(TutorialConfig.h.in TutorialConfig.h)

add_subdirectory(MathFunctions)

# add the executable
add_executable(Tutorial tutorial.cxx)

# Оказывается в прошлых версиях они не добавляля названия библиотек напрямую, а использовали list переменные, которые потом просто вставляли куда нужно
list(APPEND EXTRA_LIBS MathFunctions)
list(APPEND EXTRA_LIBS tutorial_compiler_flags)
list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions")

target_link_libraries(Tutorial PRIVATE ${EXTRA_LIBS})
# Если PRIVATE, на этапе линковки MyApp не видно MathFunctions из Tutorial-а

target_include_directories(Tutorial PUBLIC
# find TutorialConfig.h in build directory
                                        ${PROJECT_BINARY_DIR}
# Это доступ только к заголовочным файлам,
# если включаем PUBLIC возможно MyApp не сможет получить доступ к machine code MathFunctions
# Я закоментил, потому что мы добавили include requirement в MathFunctions, теперь включается по умолчанию
                                        # ${EXTRA_INCLUDES}
)

# --- Option ---

# USE_MYMATH здесь виден, хотя объявляется в subdirectory
# if (USE_MYMATH)
# pass precompiled definition to be accessible from our source files
#     target_compile_definitions(Tutorial PRIVATE USE_MYMATH)
# endif()

# Use cmake-gui or:
#$ cmake -DUSE_MYMATH=OFF|ON
# CMake will remember choice

# ------

# --- Install ---

# `install` executable and header files in standard location to be found by terminal
install(
    TARGETS Tutorial
    DESTINATION bin
)
# CMAKE_PROJECT_NAME - Global project(name)
# PROJECT_NAME - local project(name)
install(
    FILES ${PROJECT_BINARY_DIR}/TutorialConfig.h
    DESTINATION include/${CMAKE_PROJECT_NAME}
)

# Include License in packed archive
install(
	FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
	DESTINATION share/doc/${CMAKE_PROJECT_NAME}
)

#$ cmake -S .. -B .
#$ cmake --build .
#$ cmake --install .
#   --config <Debug|Release>
#   --prefix <override-directory>

# ------

# --- CTest ---

# enable_testing()
# CDash
include(CTest)

# Smoke Test, verify that app runs, does not segfault or otherwise crash
add_test(NAME Runs COMMAND Tutorial 25)
# Simply runs ./Tutorial 25 and checks if it returns 0

# Unit test?
# Check the correct behaviour with incorrect input
add_test(NAME Usage COMMAND Tutorial)
# RegEx, главное, чтобы хоть один участок подошел, можно даже оставить только "Usage:"
set_tests_properties(Usage PROPERTIES PASS_REGULAR_EXPRESSION "Usage:.*number.")

# Check that it is trully square root
add_test(NAME StandardUse COMMAND Tutorial 4)
set_tests_properties(StandardUse PROPERTIES PASS_REGULAR_EXPRESSION "4 is 2")

# function(<function-name> <args-list>)
function(do_test target arg result)
    add_test(NAME Comp${arg} COMMAND ${target} ${arg})
    set_tests_properties(Comp${arg} PROPERTIES PASS_REGULAR_EXPRESSION ${result})
endfunction()
# do a bunch of result based tests
do_test(Tutorial 4 "4 is 2")
do_test(Tutorial 9 "9 is 3")
do_test(Tutorial 5 "5 is 2.236")
do_test(Tutorial 7 "7 is 2.645")
do_test(Tutorial 25 "25 is 5")
do_test(Tutorial -25 "-25 is (-nan|nan|0)")
do_test(Tutorial 0.0001 "0.0001 is 0.01")

#$ cmake ..
#$ ctest -N|--show-only
#$ ctest -VV|--extra-verbose

# ------

include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${Tutorial_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${Tutorial_VERSION_MINOR}")
set(CPACK_GENERATOR "TGZ")
set(CPACK_SOURCE_GENERATOR "TGZ")
include(CPack)
