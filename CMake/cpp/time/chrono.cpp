#include <iostream>
#include <chrono>

int main(){
    // --- System clock
    /*
    https://en.cppreference.com/w/cpp/chrono
        - system_clock - CLOCK_REALTIME
        - steady_clock - CLOCK_MONOTONIC
    */
    auto now = std::chrono::system_clock::now();

    // time_t from <ctime>, time(0)
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::cout << "Текущее время: " << std::ctime(&now_c);

    // Duration from (1970-01-01)
    auto epoch = now.time_since_epoch();
    std::cout << "Milliseconds from epoch: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count()
            << "ms\n";

    // --- Duration ---
    /*
    duration < Representation (int or double), Period = std::ratio<1, 1> >
    По умолчанию duration в секундах

    std::ration<Num, Denominator> i.e. Num / Denominator
        std::ratio<60, 1> - 60 единиц на 1 установленную
        std::ratio<1, 1000> - 1 единица на 1000 установленных
    */
    std::chrono::duration<double> half_sec(0.5); // 0.5 секунды
    std::cout << half_sec.count() << std::endl;

    // chrono::nanoseconds : chrono::duration<int, std::nano> nano - std::ratio<1, 1*10^9> i.e. 1/10^9
    std::chrono::seconds s1(2), s2(3);
    // Arithmetic
    std::chrono::seconds sum = s1 + s2;  // 5 секунд
    // Implicit Casting ()
    std::chrono::milliseconds ms = sum;  // Расширяющее преобразование, 5000 миллисекунд
    std::chrono::nanoseconds ns = sum;   // nanoseconds
    // Explicit Casting
    // std::chrono::seconds secs = ns;   // Error!
    std::chrono::seconds secs = std::chrono::duration_cast<std::chrono::seconds>(ns); // Ok!

    /*
    // How duration_cast works

    // 5 seconds to milliseconds
    ms.count() = sec.count() * 1000;
          = 5 * 1000;
          = 5000;

    // 2500 milliseconds to seconds
    sec.count() = ms.count() / 1000;
          = 2500 / 1000;
          = 2;
    */

    std::cout << ms.count() << std::endl;
    std::cout << ns.count() << std::endl;

    return 0; // [Inferior 1 (process 25949) exited normally]
    // return 2; // [Inferior 1 (process 25921) exited with code 02]
}
