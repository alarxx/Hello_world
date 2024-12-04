# Function
Function Definition
Function Call
**Coding Principle No. 10** Make all function arguments const, except when changing value (see later).
**Coding Principle No. 11** Make sure, that a function has a call to return in every execution path.

Main Function
Call-by-value const
Call-by-reference &
Recursion
Function Naming
**Coding Principle No. 12** Functions implementing the same algorithm on different types should be named equal.

Default Arguments and Function Names
Function Name Scope
Inline Functions
Function Pointers
Functions and Minimal Evaluation (function in if statement)
**Coding Principle No. 13** Never rely on a function call in a logical expression.

Function and static Variables


# Arrays and Dynamic Memory
Array Definition
Array Access
**Coding Principle No. 14** Always make sure, that you access arrays within the valid index range. (The program will only terminate, if the memory does not belong to the program)
Array Operations (copy, add)
Multidimensional Arrays
Arrays = Pointer
Dynamic Memory
```c++
double * v = new double[ n ];
delete[] v;
```
**Coding Principle No. 15** (Dangling Pointer) After calling delete, reset the pointer value to NULL.
**Coding Principle No. 16** (Memory Leak) Always make sure, that allocated memory is deallocated after using.
Multidimensional Arrays and Mapping (row-wise, column-wise)
BLAS
string
**Coding Principle No. 17** Always ensure, that strings are terminated by ’\0’.
# Advanced Data Types

when we pass call-by-value copy of the struct are created

Example of Application - Sparse Matrices

union используется одну область памяти для нескольких переменных.

**Coding Principle No. 18** Follow a strict convention in naming new types, e.g. with special prefix or suffix.


# Modules and Namespaces

**Coding Principle No. 19** Always encapsulate your header files by an ifndef-define-endif construct.
**Coding Principle No. 20** Only if absolutely neccessary make non-const variables global.
**Coding Principle No. 21** Put module local functions into an anonymous namespace.


# Classes

**Coding Principle No. 22** Always make sure, that the C++ generated default and copy constructors behave as expected. If in doubt: implement constructors by yourself.
**Coding Principle No. 23** Make all member variables of a record private and allow read-/write-access only via member functions.
**Coding Principle No. 24** Only overload operators if necessary and reasonable.
**Coding Principle No. 25** Always make sure, that the C++ generated copy operator behaves as expected. If in doubt: implement operator by yourself.



---

# Arrays and Dynamic Memory
16
![[Pasted image 20241204080255.png]]

---
17
![[Pasted image 20241204080319.png]]

---

18
![[Pasted image 20241204080338.png]]

---

19
![[Pasted image 20241204080439.png]]

21
![[Pasted image 20241204080943.png]]




# Data Types
21
![[Pasted image 20241204102340.png]]

22
![[Pasted image 20241204102356.png]]

26
![[Pasted image 20241204102458.png]]

==**check this lecture again**==
**compressed row storage format**


---

Just interesting
![[Pasted image 20241204120928.png]]
