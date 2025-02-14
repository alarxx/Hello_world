cmake_minimum_required(VERSION 3.15)
project(App LANGUAGES CXX)

# Searches in Standard Paths
# -DCMAKE_PREFIX_PATH=<path> ?
# FindMathFunctions.cmake ?
find_package("MathFunctions" REQUIRED)

# add the executable
add_executable(App tutorial.cxx)

target_link_libraries(App PRIVATE MathFunctions)
# Если PRIVATE, на этапе линковки MyApp не видно MathFunctions из Tutorial-а

# --- Install ---
install(
    TARGETS App
    DESTINATION bin
)
# Динамические библиотеки мы обычно не можем распространять в одном загрузочном,
# а Статические по идее должны быть уже вшиты в App?
# ------

# --- CPack ---

# CPack is a rabbit hole
# - QtInstaller прикольный, устанавливает в /opt/, но это не CLI-tool, idk
# - Потестил на .deb, вроде работает
#       Discovery не показывает License
#       Устанавливает в корневые /lib/ и /bin/, не знаю на сколько это правильно
#       Shared Libraries c .deb вроде работают, но нужно указывать RPATH

include(InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "TGZ;DEB")
set(CPACK_SOURCE_GENERATOR "TGZ;DEB")

# CPACK_PACKAGE_CONTACT
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Alar Akilbekov alar.akilbekov@gmail.com")

include(CPack)

# ------
