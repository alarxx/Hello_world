#! /bin/sh

cd ./mymathlib
g++ mymath.cpp -c
ar rcs libmymath.a mymath.o

echo Static library generated!
