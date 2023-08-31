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
$ java --module-path appmodules -m appstart/appstart/mymodappdemo/MyModAppDemo
```