#include <stdio.h>
#include <stdlib.h>
#include <locale.h>

struct Person {
	int age;
	char name[256];
	float size;
};

int main(){
	// Настраиваем locale для отображения русской кодировки в терминале
	char *locale = setlocale(LC_ALL, "");

	printf("Hello, World! \n");
	printf("Привет, Мир! \n");

	// Pointers and addresses
	int a = 013; // octal, to convert try strol
	int *b = &a;
	*b = 12; // this should change a

	printf("a = %i \n", a);
	printf("&a = %p \n", &a);

	printf("*b = %d \n", *b);

	printf("b = %p \n", b);
	printf("&b = %p \n", &b);

	// Strings

	// char=1byte=8bit - ASCII=7bit
	char str[] = "String as char array \n";
	printf(str);

	// Memory Allocation 

	char *pstr = malloc(7); // memory allocation in bytes
	if(pstr == NULL){ // If space is insufficient, allocation fails and returns a NULL pointer. (geeksforgeeks)
		fprintf(stderr, "Memory allocation failed!\n"); // basically writes to a file stream
		// exit(0);
		return 1;
	}

	pstr[0] = '1';
	// Type Casting
	pstr[1] = (char) 2 + 40; // first ASCII values are commands and not printable
	pstr[2] = (char) 3l + 40l;
	pstr[3] = (char) '4';
	pstr[4] = (char) 5.f + 40.f;
	pstr[5] = (char) 6. + 40.; // double by default
	pstr[6] = (char) 7.l + 40.l; // long double
	pstr[7] = '\0'; // Null Character Byte
	printf("pstr[1] (%%c) = %c\n", pstr[1]); // I checked to see why not visible, ASCII
	printf("%s \n", pstr);

	free(pstr);

	// Type Casting
	int mynumber = 35;
	char mychar = (char) mynumber;
	// char mychar = 'A';
	int myascii = (int) mychar;
	printf("mynumber = %i \n", mynumber);
	printf("mychar = %c \n", mychar);
	printf("myascii = %i \n", myascii);


	return 0;
}

// run:  gcc -Wall hello.c -o hello.bin & ./hello.bin

/*
Output:

Hello, World!
Привет, Мир!
a = 11
&a = 0x7ffeb480d3e0
*b = 11
b = 0x7ffeb480d3e0
&b = 0x7ffeb480d3d8
String
pstr[1] (%c) = *
1*+4-./
mynumber = 35
mychar = #
myascii = 35
*/
