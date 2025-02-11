/** A simple program that builds a sqrt table
 *
 * Основная идея тут сгенерировать header файл с массивом в котором хранятся заранее посчитанные квадратные корни.
 * Такой метод оптимизации, по типу Dynamic Programming.
 *
 * Мы записываем эту "таблицу" (массив) в файл с названием передаваемым через аргумент,
 * В примере мы записываем в Table.h файл.
 * В результате в нем должно находиться объявление массива в таком виде:

  double sqrtTable[] = { 0, 1, 1.41421, 1.73205, 2, 2.23607, 2.44949, 2.64575, 2.82843, 3, 0};

  */

#include <cmath>
#include <fstream>
#include <iostream>

int main(int argc, char* argv[]){
  std::cout << "Running MakeTable.cxx" << std::endl;

  // make sure we have enough arguments
  if (argc < 2) {
    return 1;
  }

  std::ofstream fout(argv[1], std::ios_base::out);
  const bool fileOpen = fout.is_open();
  if (fileOpen) {
    // Hardcode-ят строки в файл
    // double sqrtTable[] = { 0, 1, 1.41421, 1.73205, 2, 2.23607, 2.44949, 2.64575, 2.82843, 3, 0};
    fout << "double sqrtTable[] = {" << std::endl;
    for (int i = 0; i < 10; ++i) {
      fout << sqrt(static_cast<double>(i)) << "," << std::endl;
    }
    // close the table with a zero
    fout << "0};" << std::endl;
    fout.close();
  }
  return fileOpen ? 0 : 1; // return 0 if wrote the file
}
