message("MakeTable.cmake")

add_executable(MakeTable MakeTable.cxx)

target_link_libraries(MakeTable PRIVATE tutorial_compiler_flags)

# Выполняется во время компиляции, а не генерации Makefile
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Table.h
    COMMAND MakeTable ${CMAKE_CURRENT_BINARY_DIR}/Table.h
    DEPENDS MakeTable
)
# Output тут видимо нужен, чтобы не перекомпилировать одно и то же, как target в Make
