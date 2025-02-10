message("MathFunctions CMakeLists.txt")

# add_library(MathFunctions STATIC MathFunctions.cxx mysqrt.cxx)
# but mysqrt is optional now
add_library(MathFunctions STATIC MathFunctions.cxx)
# C++ Standard Version Requirement
# Из-за PRIVATE не распространяется на Tutorial
target_link_libraries(MathFunctions PRIVATE tutorial_compiler_flags)

# Adding option
option(USE_MYMATH "Use tutorial provided math implementation" ON)

# Включаем модуль CheckCXXSourceCompiles
include(CheckCXXSourceCompiles)
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
# Добавляем и Линкуем библиотеку, если она нужна
# Порядок важен
    add_library(SqrtLibrary STATIC mysqrt.cxx)
    target_link_libraries(MathFunctions PRIVATE SqrtLibrary)
# C++ Standard Version Requirement
# Из-за PUBLIC распространяется и на MathFunctions
    target_link_libraries(SqrtLibrary PUBLIC tutorial_compiler_flags)

# Ещё один способ линковки, но тут нюанс, с PUBLIC нужно указывать absolute path, чтобы все видели исходный
# ${CMAKE_CURRENT_SOURCE_DIR}/mysqrt.cxx
# ?
    # target_sources(MathFunctions PRIVATE mysqrt.cxx)

# Почему-то добавление свойств работает после линковки библиотеки
    if(HAVE_LOG AND HAVE_EXP)
    # if(NOT HAVE_LOG AND NOT HAVE_EXP)
# Я не знаю почему здесь в виде строки "HAVE_LOG" "HAVE_EXP"
        target_compile_definitions(SqrtLibrary PRIVATE HAVE_LOG "HAVE_EXP")
    endif()
endif()

# target include requirements, now you can only link with library and include it's headers
target_include_directories(MathFunctions INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# `install` libraries and header files in standard location to be found by terminal
set(installable_libs MathFunctions tutorial_compiler_flags)
if(TARGET SqrtLibrary)
    list(APPEND installable_libs SqrtLibrary)
endif()
install(TARGETS ${installable_libs} DESTINATION lib)
install(FILES MathFunctions.h DESTINATION include)
# Сейчас flat, и по идее, нужно оборачивать библиотеку так, чтобы было:
# lib/MathFunctions/MathFunctions.a
# include/MathFunctions/MathFunctions.h
