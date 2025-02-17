# Time

---

Может быть использован для:
- [[C C++  Programming Language - Random Generator]]
- [[Frames per Second (FPS)]]

---

Epoch starts from 1970

# C

**Basic time in seconds:**
```c
#include <stdio.h> // printf
#include <time.h> // time

int main(){
    // long unsigned = time_t
    time_t t = time(NULL); // seconds from 1970
	// time(&t);
	// time(NULL), time(0)
    printf("Current time: %lu \n", t);
}
```

---

**Как получить nanoseconds?**
https://linux.die.net/man/3/clock_gettime

```c
#include <stdio.h>
#include <time.h>

int main(){

    struct timespec ts;

    // CLOCK_REALTIME (can be changed) vs. CLOCK_MONOTONIC (only grows)
    clock_gettime(CLOCK_REALTIME, &ts);

    time_t seconds = ts.tv_sec;
    // time value nanoseconds
    long nanoseconds = ts.tv_nsec; // только nano часть

    printf("Seconds: %lu \n", seconds);
    printf("Nanoseconds: %ld \n", nanoseconds); // nano = 10^9

    long total_nanoseconds = seconds * 1000000000l + nanoseconds;
    printf("Total time in nanoseconds: %ld\n", total_nanoseconds);

    return 0;
}
```

Microseconds from nanoseconds:
```c
long microseconds = ts.tv_nsec / 1000l;
long total_microseconds = seconds * 1000000l + microseconds;
```
or use timeval

---

https://www.gnu.org/software/libc/manual/html_node/Time-Types.html
- clock_t - CPU Time
- time_t - seconds since epoch (1970) - [[Coordinated Universal Time (UTC)|UTC]]
- struct timespec:
	- time_t tv_sec
	- long tv_nsec - 1.000.000.000
- struct timeval:
	- tv_sec
	- long int tv_usec - micro seconds
- struct tm

---

# C++

#### ctime
https://www.w3schools.com/cpp/cpp_date.asp

Можно использовать тот же самый метод, что и в C, только импортировать нужно будет не `<time.h>`, а `<ctime>`.
```cpp
#include <ctime> // <time.h>
...

time_t timestamp = time(NULL); // seconds
std::cout << ctime(&timestamp) << std::endl;

// --- tm ---

struct tm datetime;

datetime = *localtime(&time(NULL)); // UTC
// datetime = *gmtime(&timestamp); // GMT=UTC+1
std::cout << datetime.tm_hour << std::endl;
std::cout << asctime(&datetime) << std::endl; // "%a %b %e %H:%M:%S %Y"


char outstrf[50];
// WeekDay Month (Mon) day (space_day), 20xx (xx)
strftime(outstrf, 50, "%a %B (%b) %d (%e), %Y (%y)", &datetime);
// 12hour/minutes/seconds
strftime(outstrf, 50, "%I:%M:%S %p", &datetime);
// month/day/year
strftime(outstrf, 50, "%m/%d/%y", &datetime);


std::string weekdays[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};

std::cout << "The date is on a " << weekdays[datetime.tm_wday] << std::endl;
std::cout << "The year day is " << datetime.tm_yday << std::endl;

// -- Difference (difftime) ---

time_t now, nextyear;
now = time(NULL);

// --- Specify next year (mktime) ---
datetime = *localtime(&now);
datetime.tm_year = datetime.tm_year + 1; // Number of years since 1900
datetime.tm_mon = 0; // Number of months since January
datetime.tm_mday = 1;
datetime.tm_hour = 0; datetime.tm_min = 0; datetime.tm_sec = 0;
// Daylight Savings must be specified
// -1 uses the computer's timezone setting
datetime.tm_isdst = -1;

nextyear = mktime(&datetime);

int diff = difftime(nextyear, now); // in seconds

std::cout << diff / (24*60*60) << " days until next year \n";
```

---

#### chrono
https://cplusplus.com/reference/chrono/high_resolution_clock/now/

А еще можно использовать `<chrono>`:
```cpp
#include <iostream>
#include <chrono>

using namespace std;

int main(){
	auto now = chrono::high_resolution_clock::now();

	auto nanos =
	chrono::duration_cast<chrono::nanoseconds>(
		now.time_since_epoch
	).count();

	cout << "time since epoch in nanoseconds:" << nanos << endl;

	return 0;
}
```

---
