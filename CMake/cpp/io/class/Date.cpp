/*
ttps://learn.microsoft.com/en-us/cpp/standard-library/overloading-the-output-operator-for-your-own-classes?view=msvc-170
*/

#include <iostream>

class Date {
private:
    int month, day, year;
public:
    Date(int day, int month, int year){
        this->day = day;
        this->month = month;
        this->year = year;
    }
    friend std::ostream& operator << (std::ostream& os, const Date& date);
};

std::ostream& operator << (std::ostream& os, const Date& date){
    os << date.day << '/' << date.month << '/' << date.year;
    return os;
}

int main(){
    Date date(25, 2, 25);
    std::cout << date << std::endl;
}
