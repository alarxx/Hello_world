# Examples of modules from Herbert Schildt's book

Execute all commands from /mymodapp directory
### Flags:
```
-d - directory where the compiler will put the compiled files.

--module-path - compiler will look for modules there.

--module-source-path - to compile all necesarry files in a directory.

-m - main file in form packages.class.
```

### Compile commands:
```
$ javac -d appmodules/appfuncs appsrc/appfuncs/appfuncs/simplefuncs/SimpleMathFuncs.java
$ javac -d appmodules/appfuncs appsrc/appfuncs/module-info.java
$ javac -d appmodules/appstart appsrc/appstart/module-info.java appsrc/appstart/appstart/mymodappdemo/MyModAppDemo.java
```

### Or just compile all with this command, but delete previous build:
```
$ javac -d appmodules --module-source-path appsrc appsrc/appstart/appstart/mymodappdemo/MyModAppDemo.java
```

### Execute command:
```
$ java --module-path appmodules -m appstart/appstart.mymodappdemo.MyModAppDemo
```

### Keypoints of modules
```
module moduleName {  
    exports packageName;
    requires moduleName;
    exports packageName to moduleName;  
    
    // Indirect dependence
    requires transitive moduleName;
    
    // Services
    provides package.serviceProviderInterface 
        with package.serviceProviderImps; // comma-separated
    uses package.serviceProviderInterface;

    // Только с рефлексией
    opens packageName; // can use with to
    
    // Обязательный при компиляции, необязательный во время выполнения ?
    requires static moduleName;
}  

// Пакеты доступны во время выполнения через рефлексию,
// но на этапе компиляции доступны только явно экспортируемые пакеты
open module moduleName {}
```
Unnamed Module (without module-info.java) require all other modules
and will export all its packages as well.

### JLINK
jlink связывает модульные скомпилированные файлы в оптимизированный образ.  

Для использования jlink, jar и т.д. нужно добавить в переменные среды в PATH путь к bin java, у меня он:
"C:\Program Files\Java\jdk-20\bin"

flags
```
--launcher AppBatchName=dir/Main // compiled
--module-path 
    "%JAVA_HOME%"/jmods; // Java API
    mymodapp/appmodules  // Our modules
    
// Какие модули нужно включить, 
// jlink подхватыет нужные модули автоматически.
// Указываем модуль с точкой входа
--add-modules appstart  

--output appdirname
```

jlink example
```
$ jlink 
    --launcher MyModApp=appstart/appstart.mymodappdemo.MyModAppDemo
    --module-path "%JAVA_HOME%"/jmods;mymodapp/appmodules
    --add-modules appstart 
    --output mylinkedmodapp
```

### JAR creation example
```
--create 
--file=mymodapp/applib/name.jar

// в какую директорию и какие файлы включать (все)
-C dir .

// точка входа, если она есть
--main-class package.Main       
```
```
$ jar 
    --create 
    --file=mymodapp\applib\appstart.jar
    --main-class=appstart.mymodappdemo.MyModAppDemo
    -С mymodapp\appmodules\appstart .
```
Linking JAR files:
```
$ jlink 
    --launcher MyModApp=appstart
    --module-path "%JAVA_HOME%"/jmods;mymodapp/applib
    --add-modules appstart 
    --output mylinkedmodapp
```
Execute JAR files:
```
$ java -р mymodapp\applib -m appstart
```

### JMOD - новый формат
### jpackage - native executable applications