#! /bin/sh

cd mymathlib
# Position Independent Code (PIC), но и без этого работает почему-то
g++ mymath.cpp -c -fPIC
g++ mymath.o -o libmymath.so -shared

echo Shared library generated!

cd ../
g++ main.cpp -c
# g++ main.o -o app -lmymath -L./mymathlib -Wl,-rpath=./mymathlib
g++ main.o -o app -ldl

echo "app's shared object dependencies:"

# ldd - print shared object dependencies
ldd app

echo "Note that app doesn't depend on libmymath.so!"
echo "It will link in runtime"
