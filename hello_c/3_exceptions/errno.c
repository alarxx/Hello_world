#include <stdio.h>
#include <errno.h>
#include <string.h>

int main() {
    FILE *file = fopen("nonexistent.txt", "r");
    if (file == NULL) {
        printf("Error Number: %i\n", errno);
        printf("Error opening file. \nstrerror(errno): %s\n", strerror(errno));
    }
    return 0;
}

