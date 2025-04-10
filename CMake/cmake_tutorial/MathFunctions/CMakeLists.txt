message("MathFunctions CMakeLists.txt")

# message("PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
# message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
# message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
# message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
# message("CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

# Модульность CMake
# Чем отличается от `add_subdirectory()`?
include(MakeTable.cmake)

# add_library(MathFunctions STATIC MathFunctions.cxx mysqrt.cxx)
# but mysqrt is optional now
add_library(MathFunctions MathFunctions.cxx)
# C++ Standard Version Requirement
# Из-за PRIVATE не распространяется на Tutorial
target_link_libraries(MathFunctions PRIVATE tutorial_compiler_flags)

# Packaging Debug and Release
set_property(TARGET MathFunctions PROPERTY VERSION "1.0.0")
set_property(TARGET MathFunctions PROPERTY SOVERSION "1")

# Adding option
# -DUSE_MYMATH=OFF|ON, choice will be cached
option(USE_MYMATH "Use tutorial provided math implementation" ON)

# Включаем модуль CheckCXXSourceCompiles
include(CheckCXXSourceCompiles)
# P.S.: Я не хочу использовать реализацию exp log, но удалять код не хочу,
# set 1, чтобы использовать exp log
option(USE_EXP_LOG_OPTION "Use exp log to find the root" OFF)
if(USE_EXP_LOG_OPTION)
    set(USE_EXP_LOG 1)
else()
    set(USE_EXP_LOG 0)
endif()
message("USE_EXP_LOG: ${USE_EXP_LOG}")
if(NOT DEFINED HAVE_LOG)
# Используем функции из этого модуля для проверки компиляции кода
    check_cxx_source_compiles("
        #include <cmath>
        int main(){
            std::log(1.0);
            return 0;
        }
    " HAVE_LOG)
    message("HAVE_LOG: ${HAVE_LOG}")
endif()
if(NOT DEFINED HAVE_EXP)
    check_cxx_source_compiles("
        #include <cmath>
        int main(){
            std::exp(1.0);
            return 0;
        }
    " HAVE_EXP)
    message("HAVE_EXP: ${HAVE_EXP}")
endif()

if (USE_MYMATH)
# pass precompiled definition to be accessible from our source files (#define USE_MYMATH)
    target_compile_definitions(MathFunctions PUBLIC USE_MYMATH)
    # define the symbol stating we are using the declspec(dllexport) when building on windows
    target_compile_definitions(MathFunctions PRIVATE "EXPORTING_MYMATH")
# Добавляем и Линкуем библиотеку, если она нужна
# Порядок важен
    # add_library(SqrtLibrary STATIC
    add_library(SqrtLibrary
        mysqrt.cxx
        ${CMAKE_CURRENT_BINARY_DIR}/Table.h
    )
    # State that SqrtLibrary need PIC when the default is shared libraries
    # Почему-то работает и без этого, и вопрос почему мы другим библиотекам не пишет это
    set_target_properties(SqrtLibrary PROPERTIES
        POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS}
    )
    target_include_directories(SqrtLibrary PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(MathFunctions PRIVATE SqrtLibrary)
# C++ Standard Version Requirement
# Из-за PUBLIC распространяется и на MathFunctions
    target_link_libraries(SqrtLibrary PUBLIC tutorial_compiler_flags)

# Ещё один способ линковки, но тут нюанс, с PUBLIC нужно указывать absolute path, чтобы все видели исходный
# ${CMAKE_CURRENT_SOURCE_DIR}/mysqrt.cxx
# ?
    # target_sources(MathFunctions PRIVATE mysqrt.cxx)

# Почему-то добавление свойств работает после линковки библиотеки
    target_compile_definitions(SqrtLibrary PRIVATE USE_EXP_LOG=${USE_EXP_LOG})
    if(USE_EXP_LOG EQUAL 1 AND HAVE_LOG AND HAVE_EXP)
# Я не знаю почему здесь в виде строки "HAVE_LOG" "HAVE_EXP"
        target_compile_definitions(SqrtLibrary PRIVATE HAVE_LOG "HAVE_EXP")
    endif()
endif()

# target include requirements, now you can only link with library and include it's headers
target_include_directories(
    MathFunctions
    INTERFACE
    # Only one of these:
    # absolute path tied to the current machine
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    # export different location when being used from within the build directory and install / package.
    $<INSTALL_INTERFACE:include/${CMAKE_PROJECT_NAME}>

    # Обычно в библиотеках интерфейсы находятся в ./include/Library/, и поэтому должно бы быть так:
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    # Мы в install(FILES MathFunctions.h) указываем DESTINATION include/Tutorial/include
    # $<INSTALL_INTERFACE:include/${CMAKE_PROJECT_NAME}/include>
)

# `install` libraries and header files in standard location to be found by terminal
set(installable_libs MathFunctions tutorial_compiler_flags)
if(TARGET SqrtLibrary)
    list(APPEND installable_libs SqrtLibrary)
endif()
install(
    TARGETS ${installable_libs}
# `EXPORT` generates MathFunctionsTargets.cmake с кодом для импорта перечисленных target-ов.
    EXPORT MathFunctionsTargets
    DESTINATION lib/${CMAKE_PROJECT_NAME}
)
install(
    FILES MathFunctions.h
# По идее, нужно оборачивать библиотеку так, чтобы было:
# lib/MathFunctions/MathFunctions.a
# include/MathFunctions/MathFunctions.h
    DESTINATION include/${CMAKE_PROJECT_NAME}
)
