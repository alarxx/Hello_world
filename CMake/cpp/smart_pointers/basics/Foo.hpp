#pragma once
#ifndef _FOO_H_
#define _FOO_H_

class Foo {
public:
    Foo() { std::cout << "Foo создан" << std::endl; }
    ~Foo() { std::cout << "Foo удалён" << std::endl; }
    void show() { std::cout << "Работаю в Foo" << std::endl; }
};

#endif
