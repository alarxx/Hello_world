### SFINAE

check:
- enable_if.cpp

#### Brief about SFINAE

**SFINAE** - это Compile-time проверка подходит ли тип под специализацию шаблона.
Специализированные шаблоны которые не подходят под переданный тип исключаются и переходят на следующий или base, а не выкидывают ошибку.

Мы использум SFINAE, если хотим создать специализацию только под определенный тип и не разрешать использовать неподходящие.

- `enable_if<Condition, Type>`
- `Condition<T, expression(T) = void> : true_type | false_type` `::value`

```cpp
template <bool Condition, template Type>
class enable_if {};

// Specialization
template <typename Type>
class enable_if<true, Type> {
public:
	static constexpr type = Type;
};

template <bool Condition, template Type>
using enable_if_t = enable_if<Condition, Type>::type;
```

Для SFINAE очень важно понимание Template Specialization.
Нам не хватит одного using:
```cpp
template <typename T>
using has_size = decltype(declval<T>().size(), true_type/*false?*/);
```

`Condition`:
```cpp
// Default Base
template <typename T, typename U = void>
class has_size : false_type {};
// U нужен только для проверки T, он по итогу всегда void

// Specialization
template <typename T>
class has_size<T, decltype( ? , void())> : true_type {};

template <typename T>
using has_size_v = has_size<T>::value;
```

На месте ? мы должны вписать проверку:
```cpp
decltype(declval<T>().size(), void())
```

Using:
```cpp
template <typename T>
enable_if_t<has_size_v<T>, T>
fun(T t){...}

fun(string("Hello")); // Ok
fun(vector({1, 2, 3})); //Ok
fun(42.f); // Compile-time Error
```

---
