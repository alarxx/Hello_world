# Application structure organization

**Options:**
- folders per class
- folders per section
- pure flat
- headers for everything
	- one main cpp file
	- single object rebuild flaw (long compile time for a single change)
- monolithic - "thick", i.e. all project in one main.cpp

---

**Folder per class:**  
src/
- Data/
	- Data.cpp
	- Data.h
- main.cpp

---

### Build and Run

**Compilation:**
```sh
g++ main.cpp Data/Data.cpp -o program
```

**Running:**
```sh
./program
```
