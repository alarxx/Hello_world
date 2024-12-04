// Почему-то g++ компилирует без ошибок без Include Guards

// Include Guards
// #ifndef

// #pragma once
// Еще легче использовать, это облегчает naming-и
// Но, не знаю как будет работать, если файлы будут одинакого называться

#ifndef UTILS_H
#define UTILS_H

#include <string>

#define SQUARE(x) ((x) * (x))

// Прототип функции
std::string greet(const std::string& name);

#endif

