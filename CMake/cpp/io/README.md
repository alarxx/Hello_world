# Streams

File Handling, Stream extraction and insertion, operator << >>

---

CPP Files
https://www.w3schools.com/cpp/cpp_files.asp

```cpp
#include <iostream>
#include <fstream>
#include <string> // string, getline

int main(){
    // out file stream
    std::ofstream MyFile("file.txt");

    MyFile << "Files are tricky! \n";

    MyFile.close();

    std::string myText;

    std::ifstream MyReadFile("file.txt");

    while(getline(MyReadFile, myText)){
        std::cout << myText;
    }
    std::cout << std::endl;

    MyReadFile.close();
}
```

---

https://www.geeksforgeeks.org/file-handling-c-classes/

- Console I/O operation
- Disk I/O operation
	- `ifstream`
	- `ofstream`
	- `fstream`

Stream extraction `operator >>`
Stream insertion `operator <<`

- **`ios`**
- `istream` `:ios`
	- `operator >>`
	- `get()`
	- `getline()`
	- `read()`
- `ostream` `:ios`
	- `operator <<`
	- `put()`
	- `write()`
- `streambuf`
	- `buffer *`
- `fstreambase`
	- `open()`
	- `close()`
- `ifstream` `:istream`
	- `open()` with default input mode
	- `get()`
	- `getline()`
	- `read()`
	- `seekg()` and `tellg()`
- `ofstream` `:ostream`
	- `open()` with default output mode
	- `put()`
	- `write()`
	- `seekp()` and `tellp()`
- `fstream` `: istream, ostream`
	- simultaneous input and output operations
- `filebuf`

![[Pasted image 20250225132523.png]]

```cpp
#include <fstream>
```

```cpp
// Creation of ofstream class object
ofstream fout;
string line;
// by default ios::out mode, automatically deletes
// the content of file. To append the content, open in ios:app
// fout.open("sample.txt", ios::app)
fout.open("sample.txt");

// Execute a loop If file successfully opened
while (fout) {
	// Read a Line from standard input
	getline(cin, line);
	// Press -1 to exit
	if (line == "-1")
		break;
	// Write line in file
	fout << line << endl;
}
// Close the File
fout.close();
```

```cpp
// Creation of ifstream class object to read the file
ifstream fin;
// by default open mode = ios::in mode
fin.open("sample.txt");

// Execute a loop until EOF (End of File)
while (getline(fin, line)) {
	// Print line (read from file) in Console
	cout << line << endl;
}
// Close the file
fin.close();
```

---

https://en.cppreference.com/w/cpp/language/operators
Stream extraction and insertion

---

https://learn.microsoft.com/en-us/cpp/standard-library/overloading-the-output-operator-for-your-own-classes?view=msvc-170
```cpp
#include <iostream>

class Date {
private:
    int month, day, year;
public:
    Date(int day, int month, int year){
        this->day = day;
        this->month = month;
        this->year = year;
    }
    friend std::ostream& operator << (std::ostream& os, const Date& date);
};

std::ostream& operator << (std::ostream& os, const Date& date){
    os << date.day << '/' << date.month << '/' << date.year;
    return os;
}
```

```cpp
Date date(25, 2, 25);
std::cout << date << std::endl;
```

---

## SFINAE

[[Substitution Failure Is Not An Error (SFINAE)]]

```cpp
#include <iostream>
#include <string>
#include <type_traits>

// --- Date ---
class Date {
private:
    int day, month, year;
public:
    Date(int day, int month, int year){
        this->day = day;
        this->month = month;
        this->year = year;
    }
    friend std::ostream& operator << (std::ostream& os, const Date& date);
};

std::ostream& operator << (std::ostream& os, const Date& date){
    os << date.day << '/' << date.month << '/' << date.year;
    return os;
}
// ------

// --- is_stream_supported ---
template <typename T>
using EnableP = std::void_t<
    decltype(
        std::declval<std::ostream&>() <<
        std::declval<T>())
    >;

template <typename T, typename = void>
class is_stream_supported: public std::false_type {};

template <typename T>
class is_stream_supported<T, EnableP<T>>: public std::true_type {};
// ------

// --- fun ---
template <typename T>
std::enable_if_t<is_stream_supported<T>::value, void>
fun(T t){
    std::cout << t << std::endl;
}

template <typename T>
std::enable_if_t<!is_stream_supported<T>::value, void>
fun(T t){
    std::cout << "Stream not supported!" << std::endl;
}
// ------

class A {};

int main(){
    Date date(25, 2, 25);
    // std::cout << date << std::endl;
    fun(date);

    int num = 42;
    fun(num);

    std::string str = "Hello";
    fun(str);

    A a;
    fun(a);
}
```
