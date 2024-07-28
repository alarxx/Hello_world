#include <stdio.h>
#include <stdlib.h>

int main(){
	printf("-----------------\n");
	printf("sizeof(int) = %lu \n", sizeof(int));
	printf("-----------------\n\n");

	int a[3]; // 12 bytes = 4 bytes * 3
	// a - pointer of type int (*)[n]
	// (int *) a - pointer to first element of array, equal to &a[0]

	// Initialize

	a[0] = 12345;
	// Pointer Arithmetic
	// Adding 1 computes pointer to the next value by adding sizeof(X) for type X
	*(a + 1) = 56789; // a[1]
	*(a + 2) = 54321; // a[2]

	// а - это весь массив
	printf("sizeof(a) = %lu \n\n", sizeof(a));
	// I can't prove that this is a whole array type.
	// There is no way in C to print the typeof.

	// &a - указатель, указывает на весь массив
	// Type of &a is int(*)[n]
	printf("&a = %p \n", &a);
	// Любой указатель весит 8 байт

	printf("sizeof(&a) = %lu \n", sizeof(&a)); // указатель на весь массив, type int(*)[n]
	printf("sizeof(*(&a)) = %lu , it's dereferencing of the a\n\n", sizeof(*(&a))); // a = *(&a)

	/*
		(int *) a = &a[0]

		Когда мы cast-им массив к указателю (int *) a,
		то получается указатель на первый элемент &a[0]

	 **/
	printf("sizeof((int *) a) = %lu \n", sizeof((int *) a)); // указатель на первый элемент &a[0]
	printf("sizeof(&a[0]) = %lu \n\n", sizeof(&a[0]));

	// Let's check that the values are the same by dereferencing
	printf("value of *((int *) a) = %i \n", *( (int *) a ));
	printf("value of *(&a[0]) = %i \n\n", *(&a[0]));

	printf("-----------------\n\n");

	int * b[3]; // array of pointers

	int c[3];
	int (*d)[3]; // d is a pointer to an array of 3 integers, useless basically
	// d = malloc(2 * sizeof(int));
	d = &c;
	*d[0] = 123;
	*d[1] = 456;

	printf("sizeof(b) = %lu \n\n", sizeof(b)); // 3 * 8 bytes = 24 bytes
	// We can't really prove the type of the b, that it is an array.

	printf("sizeof(c) = %lu \n", sizeof(c)); // 3 * 8 bytes = 24 bytes
	printf("c[0] = %i \n\n", c[0]);

	printf("sizeof(d)) = %lu \n", sizeof(d)); // pointer always weighs 8 bytes
	printf("d = %p \n", d); // we can check the reference value of the pointer
	printf("*d[0] = %i \n\n", *d[0]); // and the value in array by dereferencing


	return 0;
}

// run:  gcc -Wall source.c -o source.bin & ./source.bin


