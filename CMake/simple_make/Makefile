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
