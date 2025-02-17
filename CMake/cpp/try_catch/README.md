# Error handling

https://www.w3schools.com/cpp/cpp_exceptions.asp

```cpp
try {
    int age = 16;
    if(age < 18){
        // or you can throw error number, i.g. 505
        throw (age);
    }
    std::cout << "Access granted" << std::endl;
}
catch (int age){
    std::cout << "Access denied \n";
    std::cout << "Age is: " << age << std::endl;
}
```
