# Runtime Linking

**_`build.sh`_**:
```sh
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

```

---

**_`main.cpp`_**:
```cpp
#include <dlfcn.h>

int main(){
	// загружаем in runtime, а не во время линковки
	void * handle = dlopen("./mymathlib/libmymath.so", RTLD_LAZY);
	...
	// Это просто указатель на функцию
	using add_t = int (*)(int, int);
	// dlsym() search for the named symbol
	// check name mangling in call_c_from_cpp
	add_t add = (add_t) dlsym(handle, "add");
	// Usage
	int c = add(1, 1);
}
```

**Dynamic vs. Runtime Linking**

Когда мы указывали `-lmymath` линкер на этапе компиляции находил `libmymath.so` и разрешал все "symbols".

**Name mangling**

`extern "C"` заставляет компилятор C++ использовать чистое имя, без name mangling, поэтому мы можем искать функцию через `dlsym(handle, "add");`.

Check: https://github.com/alarxx/Hello_world/tree/main/CMake/call_c_from_cpp

---

Проверь, что app не слинкован никак с libmymath.so:
```sh
$ ldd ./app

       linux-vdso.so.1 (0x00007ffcaa5f0000)
       libstdc++.so.6 => /lib/...
       libc.so.6 => /lib/...
       libm.so.6 => /lib/...
       libgcc_s.so.1 => /lib/...
```

---

**Can we link in runtime static library? No**

```sh
#! /bin/sh

cd ./mymathlib
g++ mymath.cpp -c
ar rcs libmymath.a mymath.o

echo Static library generated!
```

Я попробовал слинковаться in runtime со статичной библиотекой, и коненчно не получилось
```sh
./app
# Failed to load library: ./mymathlib/libmymath.a: invalid ELF header
```
