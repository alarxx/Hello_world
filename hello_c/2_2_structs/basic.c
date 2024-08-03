#include <stdio.h>
#include <stdlib.h>

struct Person { // 264 bytes
	int age; // 4 bytes
	char name[256]; // 256 bytes
	float height; // 4 bytes
} person;


typedef struct Person Person;


int main(){
    struct Person p[4] = {
		{1, {'A'}, .123f},
		{},
		{}
	};

    printf("sizeof Person %lu \n", sizeof(person));
    printf("sizeof p %lu \n", sizeof(p));

	Person p1 = p[0];
	printf("sizeof p1 %lu \n", sizeof(p1));
	printf("sizeof p1.name %lu \n", sizeof(p1.name));

	struct Person p2 = p[1];
	printf("sizeof p2 %lu \n", sizeof(p2));
	printf("sizeof p2.name %lu \n", sizeof(p2.name));

	Person alar = (Person) {
		.age = 20,
		.height = 1.85,
		.name = {'A', 'L', 'A', 'R'}
	};
}
