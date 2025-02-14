include("../release/CPackConfig.cmake")

set(CPACK_INSTALL_CMAKE_PROJECTS
# Build directory, Project Name, Project Component, Directory
    "debug;Tutorial;ALL;/"
    "release;Tutorial;ALL;/"
# Stupidly все файлы генерятся в root директорию проекта
)

#$ mkdir debug release
#$ cd debug
#$ cmake -DCMAKE_BUILD_TYPE=Debug ..
#$ cmake --build .
#$ cd release
#$ cmake -DCMAKE_BUILD_TYPE=Release ..
#$ cmake --build .
#$
#$ cpack --config MultiCPackConfig.cmake

# Это скомпилирует проект в Debug и в Release.
# При Debug идет postfix d: MathFunctions.so -> MathFunctionsd.so
# После мы упаковываем и Debug, и Release вместе.
# Я не знаю зачем это делать, мб при тестировании?.
#
# Кстати, postfix d похож на daemon, я не уверен что это хороший postfix.
