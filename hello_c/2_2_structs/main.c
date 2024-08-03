#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Определение структуры Person
struct Person {
    int age;
    char name[256];
    float height;
    void (*hello)(void *);
};

// Функция для создания нового объекта Person
struct Person * createPerson(int age, const char* name, float height) {
    struct Person* newPerson = (struct Person*)malloc(sizeof(struct Person));
    if (newPerson == NULL) {
        printf("Ошибка выделения памяти!\n");
        exit(1);
    }
    newPerson->age = age;
    strcpy(newPerson->name, name);
    newPerson->height = height;
    return newPerson;
}

// Функция для вывода информации о человеке
void printPerson(struct Person* person) {
    if (person != NULL) {
        printf("Name: %s\n", person->name);
        printf("Age: %d\n", person->age);
        printf("Height: %.2f\n", person->height);
    }
}

void hello(void * self){
    struct Person * p = (struct Person *) self;
    printf("Hello, my name is %s \n", (*p).name);
}

int main() {
    int n;
    printf("Введите количество людей: ");
    scanf("%d", &n);

    // Динамическое выделение памяти для массива структур
    struct Person * people = (struct Person *) malloc(n * sizeof(struct Person));
    if (people == NULL) {
        printf("Ошибка выделения памяти!\n");
        return 1;
    }

    // Инициализация массива структур
    for (int i = 0; i < n; i++) {
        int age;
        char name[256];
        float height;

        printf("Введите имя: ");
        scanf("%s", name);
        printf("Введите возраст: ");
        scanf("%d", &age);
        printf("Введите рост: ");
        scanf("%f", &height);


        /*
            Здесь проблемка, что мы в двух местах делаем allocate, но только в одном?

            people = allocate(n * 16);
            P = allocate(16);
            Именно сам P мы кладем в массив:
            [[P], [], []]

            Создается копия объекта P?

            Правильным решением было бы создать массив pointer-ов:

                struct Person * people = (struct Person *) malloc(n * sizeof(struct Person *));

        */

        // int * ptr = malloc
        // ptr gives address of allocated memory
        // *ptr gives the value itself
        struct Person * tmp = (struct Person *) createPerson(age, name, height); // memory address
        people[i] = *tmp;
        people[i].hello = hello; //&print
        // free(tmp);

        printf("sizeof(people[i]) = %lu \n", sizeof(people[i]));
        printf("sizeof(*tmp) = %lu \n", sizeof(*tmp));

        printf("&people[i] = %p \n", &(people[i])); // разные почему-то
        printf("&(*tmp) = %p \n", &(*tmp));

        printf("&(people[i].name) = %p \n", &(people[i].name) ); // разные почему-то
        printf("(people[i].name) = %s \n", (people[i].name) ); // разные почему-то
        printf("&(tmp->name) = %p \n", &(tmp->name) );
        printf("(tmp->name) = %s \n", (tmp->name) );
        // Получается данные копируются?

        // printf("people[i] = %p \n", people[i]);
        printf("&people[i] = %p \n", &people[i]);
        printf("&(*tmp) = %p \n", &(*tmp));
        printf("tmp = %p \n", tmp);
        printf("&tmp = %p \n", &tmp);

        // char tmp_name[256] = "Temporary";
        // strcpy(tmp->name, tmp_name);


        // people[i].hello(&people[i]);
        // tmp->hello(&tmp);
    }

    // Вывод информации о каждом человеке
    for (int i = 0; i < n; i++) {
        (*(people + i)).hello(people + i);
        printPerson(&people[i]);
    }

    // Освобождение выделенной памяти
    free(people);
    people = NULL; // Dangling Pointer

    return 0;
}
