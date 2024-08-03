#include <stdio.h>
#include <stdlib.h>

int main() {
    // Пример использования malloc
    int *arr1 = (int *)malloc(5 * sizeof(int));
    if (arr1 == NULL) {
        printf("Ошибка выделения памяти с помощью malloc\n");
        return 1;
    }
    for (int i = 0; i < 5; i++) {
        arr1[i] = i + 1;
    }
    printf("Массив arr1: ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", arr1[i]);
    }
    printf("\n");

    // Освобождение памяти, выделенной с помощью malloc
    free(arr1);

    // Пример использования calloc
    int *arr2 = (int *)calloc(5, sizeof(int));
    if (arr2 == NULL) {
        printf("Ошибка выделения памяти с помощью calloc\n");
        return 1;
    }
    printf("Массив arr2 (инициализирован нулями): ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", arr2[i]);
    }
    printf("\n");

    // Пример использования realloc
    arr2 = (int *)realloc(arr2, 10 * sizeof(int));
    if (arr2 == NULL) {
        printf("Ошибка выделения памяти с помощью realloc\n");
        return 1;
    }
    for (int i = 5; i < 10; i++) {
        arr2[i] = i + 1;
    }
    printf("Массив arr2 после realloc: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", arr2[i]);
    }
    printf("\n");

    // Освобождение памяти, выделенной с помощью calloc/realloc
    free(arr2);

    return 0;
}
