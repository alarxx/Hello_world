# Vector Class and Scope

Тут я написал Makefile, который охватывает все .cpp и зависимости .h в виде .d, а так же рассматривается scope переменных: локальный, глобальные; extern, inline, namespace, nested namespace, friend, operator overloading. 

Если описывать одним главным принципом то, что я понял:

**Объявлять можно множество раз, а определение должно быть всего одно**. 

Ну, с одним нюансом, объявлять можно множество раз только с указанием `extern`, но для функций можно опустить, потому что функции по умолчанию extern.

Я реализовал class Vector в стиле PyTorch использования, то есть `torch::nn::function()` с помощью nested namespaces A::B {}, как A { B{ } }, использовал инкапсулированное динамическое выделение памяти внутри класса, создал конструкторы, деструкторы, copy constructor, а также переопределил операторы: copy operator =, unary operator *=, который меняет this , и binary friend operator *, который friend функция и не имеет this, то есть должен копировать и возвращать сам результирующий объект (но не ссылку!).

Binary operator overloading имеет нюансы, первое это friend функция, второе результирующий объект мы создаем ведь в Stack памяти и она должна освободиться после завершения функции, поэтому там может два раз копирование происходить один раз explicitly, второй раз implicitly (see Return Value Optimization).

А также реализовал friend copy функцию, потому что такой функционал копирования нужен в copy constructor и copy operator =.

Дополнительно намекнул на использование function pointers и Higher Order Function.

----

Main Idea:
**Multiple Declaration and One Definition**

Всё, особенно с `extern`, можно объявлять сколько угодно раз, но определение должно быть всего одно.
Объявления пишут в .h файлах, как интерфейс, который будет объявляться кучу раз, а определение в .cpp файле, который вызовется только при компиляции и войдет в слинкованный исполняемый файл как единственное определение интерфейса.

Почему функции с определением в .h файлах должны быть `inline`?
Препроцессор вставляет содержимое в .cpp, получится что у нас одинаковая функция определяется в каждом файле, который его включает.

---

Синтаксис C и C++ очень похож. C++ - это надстройка над C и язык поэтому сложнее, чем C. Дополнительный синтаксис делал кашу у меня в голове и я решил чуть разобраться и C, и с C++.

Основные моменты:
- `new`, `delete` and `delete[]` vs. `malloc` and `free()`
- call-by-reference&
- struct methods vs. function pointers
- constructors and (virtual) destructors
- `virtual` - `override` (polymorphism)
- classes:
- В `class` все переменные по умолчанию private, а в `struct` public.
- `inline` - без вызова функции в Stack
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

