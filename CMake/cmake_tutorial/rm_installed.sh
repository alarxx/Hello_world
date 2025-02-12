#! /bin/sh

# Удалить файлы созданные после `cmake --install .` и после установки с QtInstaller
# Эта штука не работает иногда, idk

cd /opt/ && rm -rf ./Tutorial_1.1

cd /bin/ && rm Tutorial
cd /lib/ && rm -rf ./Tutorial

cd /usr/local/bin/ && rm -rf Tutorial
cd /usr/local/lib/ && rm -rf ./Tutorial
