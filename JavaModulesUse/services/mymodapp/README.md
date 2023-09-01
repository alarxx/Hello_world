### Compile:
```
$ javac 
    -d appmodules 
    --module-source-path appsrc 
    appsrc/appstart/appstart/mymodappdemo/MyModAppDemo.java 
    appsrc/userfuncsimp/module-info.java
```
Стоит отметить, что мы здесь компилируем два файла, 
где один главный класс и один класс сервис провайдера. 
Так как главный класс не затрагивает провайдера на 
этапе компиляции напрямую, мы должны компилировать 
провайдера отдельно.

### Execute:
```
$ java 
    --module-path appmodules 
    -m appstart/appstart.mymodappdemo.MyModAppDemo
```