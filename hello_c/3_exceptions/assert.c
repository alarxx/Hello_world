#include <assert.h>

void process_data(int value) {
    assert(value >= 0); // Проверка, что значение не отрицательное
    // Обработка данных...
}

int main() {
    process_data(-1); // Это вызовет ошибку
    return 0;
}

