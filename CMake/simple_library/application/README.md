# Application

Здесь мы создаем исполняемый файл, который пытаемся слинковать с mymathlib. 

До этого мы должны были build библиотеку локально, check _`../mymathlib/`_.

Если мы сбилдили библиотеку, то ее можно использовать указав relative path в _`CMakeLists.txt`_.
Но, также можно "установить" библиотеку в local standard location, и тогда компилятор будет иметь доступ, сможет найти и подключить её без указания пути.

## Build and Run

Build:
```sh
mkdir build
cd build
cmake ..
make
```

Run:
```sh
./app
```
