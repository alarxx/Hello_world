#include <iostream>

void test(int x) {
    int y = x * 2;
    std::cout << "y = " << y << std::endl;
}

int main() {
    int a = 5;
    test(a);

    int arr[5] = {1, 2, 3, 4, 5};

    return 0;
}

/*

Compile:
```sh
g++ main.cpp -g
```

```sh
gdb ./a.out
```

GDB commands:
- `break <function>`
- `break <file.cpp>:<line>`
- `run`
- `next`
- `step` - call function
- `continue`
- `backtrace` or `bt`
- `print <variable>`
- `display <variable>` - отслеживание, можно посмотреть элементы массива, например.
- `info` of `breakpoints`, `locals`
- `quit` or `q`

 */
