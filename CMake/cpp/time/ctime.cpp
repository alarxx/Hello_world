// https://www.w3schools.com/cpp/cpp_date.asp
/**
 * https://www.gnu.org/software/libc/manual/html_node/Time-Types.html
 * - clock_t - CPU Time
 * - time_t - seconds since epoch (1970) - [[Coordinated Universal Time (UTC)|UTC]]
 * - struct timespec:
 *      - time_t tv_sec
 *      - long tv_nsec - 1.000.000.000
 * - struct timeval:
 *      - tv_sec
 *      - long int tv_usec - micro seconds
 * - struct tm
 */

#include <iostream>
#include <string>
#include <ctime>

int main(){
    time_t timestamp = time(NULL); // seconds
    // time(&timestamp);

    std::cout << timestamp << std::endl;
    std::cout << ctime(&timestamp) << std::endl;

    // --- tm ---

    struct tm datetime;

    timestamp = time(NULL);
    datetime = *localtime(&timestamp); // UTC
    // datetime = *gmtime(&timestamp); // GMT=UTC+1
    std::cout << datetime.tm_hour << std::endl;
    std::cout << datetime.tm_year << std::endl; // Number of years since 1900
    std::cout << asctime(&datetime) << std::endl; // "%a %b %e %H:%M:%S %Y"

    char outstrf[50];
    strftime(outstrf, 50, "%a %B (%b) %d (%e), %Y (%y)", &datetime); // WeekDay Month (Mon) day (space_day), 20xx (xx)
    std::cout << outstrf << std::endl;
    strftime(outstrf, 50, "%I:%M:%S %p", &datetime); // 12hour/minutes/seconds
    std::cout << outstrf << std::endl;
    strftime(outstrf, 50, "%m/%d/%y", &datetime); // month/day/year
    std::cout << outstrf << std::endl;
    std::cout << std::endl;

    // ---

    // --- Specify ---

    datetime.tm_year = 2023 - 1900; // Number of years since 1900
    datetime.tm_mon = 11; // Number of months since January
    datetime.tm_mday = 17;
    datetime.tm_hour = 12;
    datetime.tm_min = 30;
    datetime.tm_sec = 1;
    // Daylight Savings must be specified
    // -1 uses the computer's timezone setting
    datetime.tm_isdst = -1;

    timestamp = mktime(&datetime); // can correct errors like 32nd month

    std::cout << ctime(&timestamp);

    std::string weekdays[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};

    std::cout << "The date is on a " << weekdays[datetime.tm_wday] << std::endl;
    std::cout << "The year day is " << datetime.tm_yday << std::endl;


    // -- Difference ---

    time_t now, nextyear;
    now = time(NULL);

    datetime = *localtime(&now);
    datetime.tm_year = datetime.tm_year + 1;
    datetime.tm_mon = 0;
    datetime.tm_mday = 1;
    datetime.tm_hour = 0; datetime.tm_min = 0; datetime.tm_sec = 0;
    datetime.tm_isdst = -1;

    nextyear = mktime(&datetime);

    int diff = difftime(nextyear, now);

    std::cout << diff << " seconds until next year \n";
    std::cout << diff / (24*60*60) << " days until next year \n";
}
