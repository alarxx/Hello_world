# Using MathFunctions library

Я тестировал с использованием динамических библиотек, поэтому там некоторые запарки с RPATH,
с использованием статических библиотек можно не думать про RPATH, просто использовать библиотеку и статически линковаться с ней.

при `cmake --install .` все отлично, исполняемый файл находит динамические библиотеки в `/usr/local/lib/Tutorial/`.

Но проблема при cpack (DEB), он скачивает в головной,
и при `apt install ./.deb` должен указываться на головной `/usr/lib/`,
поэтому я вручную через терминал указывал RPATH.

Для тестирования, я сначала конечно установил основной cmake_tutorial, в котором export MathFunctions:
```sh
# Без указания RPATH DEB будет указывать в локал (либо никуда)
cmake -DCMAKE_INSTALL_RPATH="/lib/Tutorial" ..
cmake --build .
# Local
./Tutorial
# Then packing
cpack
apt remove tutorial
apt install ./Tutorial*.deb
# Global
Tutorial
```
Тут мы RPATH указываем не только для Tutorial, но он будет распространяться и дальше, для тех кто линкуется с библиотеками.

После скомпилировал и исполнил этот проект:
```sh
cmake ..
cmake --build .
./App

ldd ./App
    libMathFunctions.so => /usr/lib/Tutorial/libMathFunctions.so
    libSqrtLibrary.so => /lib/Tutorial/libSqrtLibrary.so
    ...
```

Создадим загрузчик:
```sh
# Но вот снова DEB забивает на пути к библиотеке, поэтому приходится указывать RPATH
cmake -DCMAKE_INSTALL_RPATH="/lib/Tutorial" ..
cmake --build .
# Then packing
cpack
apt remove tutorial
apt install ./Tutorial*.deb
./App
App
```

И для ./App и для App:
```sh
ldd App
    libMathFunctions.so => /lib/Tutorial/libMathFunctions.so
    libSqrtLibrary.so => /lib/Tutorial/libSqrtLibrary.so
    ...
```

Если попробовать скачать через DEB без указания RPATH, то App не будет знать где находятся динамические библиотеки.
