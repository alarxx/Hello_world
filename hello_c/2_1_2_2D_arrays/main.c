#include <stdio.h>
#include <stdlib.h>

// Функция для выделения памяти для двумерного массива
int** allocate2DArray(int rows, int cols) {
    int **array = (int **)malloc(rows * sizeof(int *));
    for (int i = 0; i < rows; i++) {
        array[i] = (int *)malloc(cols * sizeof(int));
    }
    return array;
}

// Функция для освобождения памяти двумерного массива
void free2DArray(int **array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

int main() {
    // // Статический двумерный массив размером 3x4
    // int array[3][4] = {
    //     {0, 1, 2, 3},
    //     {4, 5, 6, 7},
    //     {8, 9, 10, 11}
    // };

    int rows = 3, cols = 4;

    // Выделение памяти для двумерного массива
    int **array = allocate2DArray(rows, cols);

    // Инициализация двумерного массива
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            array[i][j] = i * cols + j;
        }
    }

    // Вывод двумерного массива
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", array[i][j]);
        }
        printf("\n");
    }

    // Освобождение памяти
    free2DArray(array, rows);

    return 0;
}

