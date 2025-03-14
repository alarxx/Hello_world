# target [target2 ...]: [prerequisites ...]
	# [command1
	#  command2
	# ........]

# Phony targets are not files
.PHONY: build \
	run \
	clean \
	help \
	print_sources \
	print_headers \
	print_objects \
	fyi_file

help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... build  	- Build"
	@echo "... run  	- Build and Run"
	@echo "... clean 	- Remove build/"
	@echo "..."
	@echo "... FYI:"
	@echo "... 	print_sources"
	@echo "... 	print_headers"
	@echo "... 	print_objects"
	@echo "... 	fyi_file"

# --- User Defined ---
# := - simple fixed assignment
# = - recursive, вычисляется каждый раз
CC = g++
CFLAGS = -Wall -pedantic
PROJECT_BINARY_DIR = out/build
TARGET = app
# ------

# 1. hard coded source files:
# 	SRCS = main.cpp scum.cpp
# 2. flat search:
# 	SRCS := ${wildcard */*.cpp}
# 3. recursive search using bash: (current approach)
# flags: -type f|d -name "" -or -and -exec -delete etc.
# выдает относительные пути в виде './dir/entry'
SRCS := $(shell find ./ -type f -name "*.cpp" -or -name "*.cxx" -or -name "*.c")
HEADERS := $(shell find ./ -type f -name "*.h" -or -type f -name "*.hpp")
# delete './':
SRCS := $(patsubst ./%, %, $(SRCS))
HEADERS := $(patsubst ./%, %, $(HEADERS))
# $(patsubst PATTERN, REPLACEMENT, TEXT)

# dir/any.cpp -> build/dir/any.o
OBJS := ${SRCS:%.cpp=${PROJECT_BINARY_DIR}/%.o}

# rebuild object files if dependency header files modified
# include makefiles, the "-" ignores errors
-include $(OBJS:.o=.d)

# prints all your source files
print_sources:
	@echo ${SRCS}
print_headers:
	@echo ${HEADERS}
# almost the same, but object files are in build/ directory
print_objects:
	@echo ${OBJS}
# FYI, make автоматически убирает './' перед и '/' после (path handling)
# Но, ${PROJECT_BINARY_DIR}/%.o будет выдавать build/./%.o
fyi_file: ./file.txt/
	@echo phony fyi_file $^
./file.txt/:
	@echo file $@

TARGET_LOCATION = ${PROJECT_BINARY_DIR}/${TARGET}

build: ${TARGET_LOCATION}
# Due to using @ echo command not displayed
	@echo Build successful!

# - `$@`: the target filename.
# - `$*`: the target filename without the file extension.
# - `$<`: the first prerequisite filename.
# - `$^`: the filenames of all the prerequisites, separated by spaces, discard duplicates.
# - `$+`: similar to `$^`, but includes duplicates.
# - `$?`: the names of all prerequisites that are newer than the target, separated by spaces.
${TARGET_LOCATION}: ${OBJS}
	${CC} ${CFLAGS} $^ -o $@

${PROJECT_BINARY_DIR}/%.o: %.cpp | ${PROJECT_BINARY_DIR}
# Создаём подкаталоги внутри build/
	mkdir -p $(dir $@)
# -MMD to track included header files as dependencies in corresponding .d files
	${CC} ${CFLAGS} $< -MMD -c -o $@

${PROJECT_BINARY_DIR}:
	mkdir -p ${PROJECT_BINARY_DIR}

# Build and Run
run: build
	@./${TARGET_LOCATION}

# Delete build directory
clean:
	find . -name "*.d" -delete -or -name "*.o" -delete
	rm ${TARGET_LOCATION}
	rm -r ${PROJECT_BINARY_DIR}
	@echo Clean done!
