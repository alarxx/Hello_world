#include <iostream>
#include <string>
#include <type_traits>

// --- Date ---
class Date {
private:
    int day, month, year;
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
// ------

// --- is_stream_supported ---
template <typename T>
using EnableP = std::void_t<
    decltype(
        std::declval<std::ostream&>() <<
        std::declval<T>()
    )>;

template <typename T, typename = void>
class is_stream_supported: public std::false_type {};

template <typename T>
class is_stream_supported<T, EnableP<T>>: public std::true_type {};
// ------

// --- fun ---
template <typename T>
std::enable_if_t<is_stream_supported<T>::value, void>
fun(T t){
    std::cout << t << std::endl;
}

template <typename T>
std::enable_if_t<!is_stream_supported<T>::value, void>
fun(T t){
    std::cout << "Stream not supported!" << std::endl;
}
// ------

class A {};

int main(){
    Date date(25, 2, 25);
    // std::cout << date << std::endl;
    fun(date);

    int num = 42;
    fun(num);

    std::string str = "Hello";
    fun(str);

    A a;
    fun(a);
}
