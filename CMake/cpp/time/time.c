#include <stdio.h>
#include <time.h>

int main(){
    time_t timestamp = time(NULL);
    // time(&timestamp);

    printf("seconds: %lu \n", timestamp);

    // in C++ we can display using ctime(), but here we can't
    // printf("seconds: %lu \n", time(&timestamp));

    // Human readable format using `asctime()`
    struct tm ltime = *localtime(&timestamp); // in UTC, to get GMT use `gmtime()`
    printf("time: %s \n", asctime(&ltime));
}
