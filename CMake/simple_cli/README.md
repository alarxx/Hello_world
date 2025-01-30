# Simple CLI-tool

- **add_executable**
- **install**

Можно использовать executable из build директории:

```sh
rm -r ./build
mkdir build
cd ./build

cmake -S .. -B .
make

./trim arg1 arg2
```
Output: arg1arg2

---

Можно вызывать откуда угодно, использовать как CLI-tool:
```sh
cd ./build

su -c "make install" root

trim arg1 arg2
```
Output: arg1arg2

Executable trim добавился в _/usr/local/bin/trim_.

Теперь мы можем вызывать команду `trim` с флагами откуда угодно. Это удобно, например, если мы часто работаем с файлами, читаем и делаем какие-то преобразования с ними.

---

Executable builds in Debug mode by default, rather than Release which is going to be optimized.

See: `CMAKE_BUILD_TYPE`

---

## References

- CMake Tutorial EP 2 | Libraries | Installing | Pairing with Executables | RE-DONE! https://www.youtube.com/watch?v=DDHCEE_PHOU
