#include <stdio.h>
#include <stdlib.h>

int main(){
	int a, b = 10, *c = &a;

	*c = 11;

	printf("a = %i \n", a);
	printf("b = %i \n", b);
	printf("c = %i \n", *c);

	return 0;
}

// run:
// gcc -Wall var_declaration.c -o var_declaration.bin \
// & ./var_declaration.bin


