#include <stdio.h>
#include <stdlib.h>
#include <locale.h>

int main(){
	char *locale = setlocale(LC_ALL, "");

	printf("Hello, World! \n");
	printf("Привет, Мир! \n");

	int a = 013;
	int *b = &a;

	printf("a = %i \n", a);
	printf("&a = %p \n", &a);

	printf("*b = %d \n", *b);

	printf("b = %p \n", b);
	printf("&b = %p \n", &b);

	// char=1byte=8bit - ASCII=7bit
	char str[] = "String \n";
	printf(str);

	char *pstr = malloc(7); // memory allocation in bytes
	pstr[0] = '1';
	pstr[1] = (char) 42;
	pstr[2] = (char) 43l;
	pstr[3] = (char) '4';
	pstr[4] = (char) 45.f;
	pstr[5] = (char) 46.;
	pstr[6] = (char) 47.l;
	pstr[7] = '\0'; // Null Character Byte
	printf("pstr[1] (%%c) = %c\n", pstr[1]);
	printf("%s \n", pstr);

	free(pstr);

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
