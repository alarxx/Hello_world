#! /bin/sh

cd mymathlib
# Position Independent Code
g++ mymath.cpp -c -fPIC
g++ mymath.o -o libmymath.so -shared

echo Shared library generated!

cd ../
g++ main.cpp -c
g++ main.o -o app -lmymath -L./mymathlib -Wl,-rpath=./mymathlib
# -Wl,-rpath=. - insert the path to the shared library into the binary
# ld -rpath=.

echo "app's shared object dependencies:"

# ldd - print shared object dependencies
ldd app

# если не использовали -rpath при линковке, то нужно указать путь к shared libraries,
# LD_LIBRARY_PATH=./ ./app
