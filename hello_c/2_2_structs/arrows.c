#include <stdio.h>

struct foo {
	int x;
	float y;
};

int main()
{
	struct foo var;
	struct foo* pvar;
	pvar = &var;

	var.x = 5;
	(&var)->y = 14.3;
	printf("x:%i - y:%.02f\n", var.x, (&var)->y);
	printf("x:%i - y:%.02f\n", pvar->x, pvar->y);
	printf("-----------\n");

	pvar->x = 6;
	(*pvar).y = 22.4;
	printf("x:%i - y:%.02f\n", var.x, (&var)->y);
	printf("x:%i - y:%.02f\n", pvar->x, pvar->y);

	return 0;
}

