#! /bin/sh

g++ main.cpp -c -std=c++98 -Wall -Wextra -pedantic
gcc legacy.c -c -std=c89 -Wall -Wextra -pedantic
g++ main.o legacy.o -o ./app
