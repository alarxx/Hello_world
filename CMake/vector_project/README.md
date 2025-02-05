# Vector Class and Scope

Main Idea:
**Multiple Declaration and One Definition**

Все, особенно с `extern`, можно объявлять сколько угодно раз, но определение должно быть всего одно.
Объявления пишут в .h файлах, а определение лучше написать в .cpp файле.

Почему функции с определением в .h файлах должны быть `inline`?
Препроцессор вставляет содержимое в .cpp, получится что у нас одинаковая функция определяется в каждом файле, который его включает.

Синтаксис C и C++ очень похож. C++ - это надстройка над C и язык поэтому сложнее, чем C. Дополнительный синтаксис делал кашу у меня в голове и я решил чуть разобраться и C, и с C++.

Основные моменты:
- `new`, `delete` and `delete[]` vs. `malloc` and `free()`
- call-by-reference&
- struct methods vs. function pointers
- constructors and (virtual) destructors
- `virtual` - `override` (polymorphism)
- classes:
- В `class` все переменные по умолчанию private, а в `struct` public.
- `inline` - без вызова функции
- `extern` - multiple declaration
- `constexpr`, `static_assert`, `assert` - вычисление при компиляции
- `friend` - не принадлежит классу, но имеет доступ к private полям
- `namespace` and `static` - scopes: named, global and local scopes
- `operator <op>` overloading
- OOP:
	- Encapsulation, using access modifiers: public, protected, private
	- Inheritance, using `:` and "initialization list"
	- Polymorphism, тип указателя может быть базовым классом (`virtual`, `override`)
	- Abstraction, за счет виртуальных функций
- templates
- lambda functions
- smart pointers
- containers
- auto

