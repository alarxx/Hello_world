# Make (Software)

CLI tool for building software.
Implemented in C in 1976 at Bell Labs.
GNU Make (**gmake**) is a standard implementation.

Устанавливается вместе с GCC.

Build:
Source files -> Object files -> Executable file

`.cpp` -> `.i `-> `.asm` -> `.obj` -> `.exe`

See more: [[Compilation Process]]

>`make` позволяет описать, какие файлы нужно компилировать, в каком порядке, и какие зависимости у этих файлов. Например, если ты изменил только один исходный файл, `make` пересоберёт только его и зависимые части, а не весь проект.

For programming languages:
- [[C Programming Language]]
- [[C++ Programming Language]]
- [[JavaScript (JS)]]
- [[TypeScript (TS)]]

---

**Configuration files:**
"`makefile`", "`Makefile`" or "`GNUMakefile`"

**How to write Makefilles:**
-> [[Make (Software) - Makefile]]

```c
# - `$@`: the target filename.
# - `$*`: the target filename without the file extension.
# - `$<`: the first prerequisite filename.
# - `$^`: the filenames of all the prerequisites, separated by spaces, discard duplicates.
# - `$+`: similar to `$^`, but includes duplicates.
# - `$?`: the names of all prerequisites that are newer than the target, separated by spaces.

CC = gcc

CFLAGS = -Wall -pedantic

TARGET = program.exe

# main.c library.c
SRCS = main.c

OBJS = $(SRCS:.c=.o)

REBUILDABLES = $(OBJS) ${TARGET}


# all: program.exe
all: $(TARGET)
# Due to using @ echo command not displayed
	@echo All done

# program.exe: main.o
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

# main.o: main.c
%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@


# Phony Target, as well as "all"
clean:
	rm -f $(REBUILDABLES)
	echo Clean done
```

# Usage

Run:
```sh
make
```
the same as running `make all`

Clean:
```sh
make clean
```

# Notes

**Configuration files:**
"`makefile`", "`Makefile`" or "`GNUMakefile`"

https://www3.ntu.edu.sg/home/ehchua/programming/cpp/gcc_make.html

**Syntax:**
```c
target1 [target2 ...]: [pre-req-1 pre-req-2 ...]
	[command1
	 command2
	 ......]
```

`#` - comments
`\` - перенос строки

---

**Simple example:**
```c
# Phony Target (all)
# 1) Запустится phony после вызова make или make all
# Начнет искать и выполнять пререквизиты - hello.exe
all: hello.exe

# 2)
# Снова пойдет искать пререквизиты, и если существует выполнять
hello.exe: hello.o
	 gcc -o hello.exe hello.o

# 3) Может запуститься, так как нет target-а hello.c
# Если hello.o не существует или hello.c новее, то выполнится
hello.o: hello.c
	 gcc -c hello.c

# Phony Target (clean)
clean:
	 rm hello.o hello.exe
```
Проход в глубину тут получается: 3 -> 2 -> 1

---

**Phony Targets (Artificial Targets)**
target-ы, которые не представляют файл: `all`, `clean`, `install`

**Variables**
`$(..)`:  `$(CC)`, `$(CC_FLAGS)`;
можно использовать `${...}`

**Virtual Path**

`VPATH = src include` - для поиска директорий

`vpath %.c src` - ищет .c файлы в src директории
`vpath %.h include` - ищет .h файлы в include директории

**Automatic Variables**
Благодаря им возможны **Pattern Rules**.
- `$@`: the target filename.
- `$*`: the target filename without the file extension.
- `$<`: the first prerequisite filename.
- `$^`: the filenames of all the prerequisites, separated by spaces, discard duplicates.
- `$+`: similar to `$^`, but includes duplicates.
- `$?`: the names of all prerequisites that are newer than the target, separated by spaces.

**Pattern Rules**
Прикольная штука, по типу шаблона.
`%` - file name without extension
```c
# Applicable for create .o object file.
# '%' matches filename.
# $< is the first pre-requisite
# $(COMPILE.c) consists of compiler name and compiler options
# $(OUTPUT_OPTIONS) could be -o $@
%.o: %.c
	$(COMPILE.c) $(OUTPUT_OPTION) $<

# Applicable for create executable (without extension) from object .o object file
# $^ matches all the pre-requisites (no duplicates)
%: %.o
$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@
```

