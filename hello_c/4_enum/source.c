#include<stdio.h>

enum week{
    Mon=123,
    Tue,
    Wed=9,
    Thur,
    Fri,
    Sat,
    Sun
};

int main()
{
	enum week day = Tue;
	printf("day (Tue) = %i \n", day);


    day = Thur;
	printf("day (Thur) = %i \n", day);

	return 0;
}
