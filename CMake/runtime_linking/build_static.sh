#! /bin/sh

cd ./mymathlib
g++ mymath.cpp -c
ar rcs libmymath.a mymath.o

echo Static library generated!

# Я попробовал слинковаться in runtime со статичной библиотекой, и коненчно не получилось
# Failed to load library: ./mymathlib/libmymath.a: invalid ELF header
