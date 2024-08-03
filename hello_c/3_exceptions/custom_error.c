#include <stdio.h>

// Псевдоним для struct Error
typedef struct {
    int code;
    const char *message;
} Error;

Error divide(int a, int b, int * result) {
    if (b == 0) {
        return (Error){.code = -1, .message = "Division by zero"};
    }
    *result = a / b;

    // C99 инициализация структур с именованными полями
    Error err = (Error){.code = 0, .message = "Success"};

    return err;
}

int main() {
    int result;
    Error error = divide(10, 1, &result);
    if (error.code != 0) {
        printf("Error: %s\n", error.message);
    } else {
        printf("Result: %d\n", result);
    }
    return 0;
}

