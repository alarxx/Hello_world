#include <pybind11/pybind11.h>

int add(int a, int b) {
    return a + b;
}

// Первый аргумент - это имя модуля, оно должно совпадать с названием скомпилированной библиотеки
PYBIND11_MODULE(example, m) {
    m.def("add", &add, "A function that adds two numbers");
}

/*
Install GCC:

    apt install build-essential

Install Python:

    apt install python3-full

Create virtual environment:

    python3 -m venv ./venv

Activate virtual environment:

    source ./venv/bin/activate

Compile:

    g++ -O3 -Wall -shared -std=c++11 -fPIC \
    `python3 -m pybind11 --includes` \
    module.cpp -o example`python3-config --extension-suffix`

Call from Python:

    python3 main.py

 */
