---
creation_time: 2024-07-08 01:33
---
[[Python]]

Не нужно делать export, как в JS. 
Для инкапсуляции можно использовать классы ([[Python OOP]]).

```python
# my_package/module.py 
def my_function():
	pass
```

**Packet** - это директория с модулями и файлов `__init__.py` (может быть пустым)
```
my_package/ 
	__init__.py 
	module.py 
```

Как использовать:
```python
from my_package import module
# module.my_function
```
or 
```python
from my_package.module import my_function
# my_function
```

---

-> How to upload?
https://packaging.python.org/en/latest/tutorials/packaging-projects/