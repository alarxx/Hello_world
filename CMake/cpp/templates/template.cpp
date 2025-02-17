#include <iostream>
#include <initializer_list>

// --- Template Function ---

// typename is equivalent to a class, but not always
template <typename T> T square(T v){
    return v * v;
}

// Template Specialization: for char data type
template <> char square<char>(char v){
// template <> char square(char v){ // Необязательно уточнять <char>
    return 'N';
}

// ------

// --- Data ---

template <typename T> class Data {
private:
    T __theData;
public:
    Data(){}
    Data(T data) : __theData(data) {}
    void set(T data){ __theData = data; }
    T get(){ return __theData; }
};

// ------

// --- Array (type , size) ---

template <typename T = int, int tsize = 0> class Array {
private:
    int __size;
    T * __coeffs;
public:
    Array();
    Array(T * array);
    Array(std::initializer_list<T> list);
    ~Array(){ delete[] __coeffs; }
    int size() const { return __size; }
    T * data(){ return __coeffs; }
    void set(int i, T x) { __coeffs[i] = x; }
    int get(int i) const { return __coeffs[i]; }
    void print();
};

// Constructor
template <typename T, int tsize> Array<T, tsize>::Array(){
    this->__size = tsize;
    this->__coeffs = new T[tsize];
}

template <typename T, int tsize> Array<T, tsize>::Array(T * array){
    __size = sizeof(array) / sizeof(array[0]); // ??
    this->__coeffs = array;
}

template <typename T, int tsize> Array<T, tsize>::Array(std::initializer_list<T> list){
    //  На случай если мы не передадим tsize (Argument Deduction)
    __size = tsize ? tsize : list.size();
    __coeffs = new T [list.size()];

    // ---
    // const int * array = list.begin(); // мы могли бы так сделать, но ...
    // 1) сами значения array [i] были бы const
    // 2) initializer_list in Static Memory, so it will be deleted after function completion what will lead to dangling pointer
    // ---
    // So we create copy
    int i = 0;
    // auto
    // for(auto it = list.begin(); it != list.end(); it++, i++){
    // std::initializer_list<T>::iterator // Тут почему-то нельзя использовать из-за шаблонного initializer_list
    // typename initializer_list<int>::iterator
    for(typename std::initializer_list<T>::iterator it = list.begin(); it != list.end() /*&& i < list.size()*/; it++, i++){
        __coeffs[i] = *it;
    }
}

template <typename T, int tsize> void Array<T, tsize>::print() {
    for(int i=0; i<this->__size; i++){
        std::cout << this->__coeffs[i] << " ";
    }
    std::cout << std::endl;
}

// ------

template <class T, int max> int arrMin(T arr[], int n){
    // stupid function, may return max
    int m = max;
    // never runs if n = 0
    for (int i = 0; i < n; i++)
        if (arr[i] < m)
            m = arr[i];
    return m;
}

// ------

int main(){
    {
        // auto sqrtval = square(5.5);
        auto sqrtval = square('A');
        std::cout << "sqrtval: " << sqrtval << std::endl;
    }
    // ------
    {
        Data<int> dataObj1;
        dataObj1.set(101);
        std::cout << "data object 1: " << dataObj1.get() << std::endl;
        // Argument Deduction
        Data dataObj2(202);
        std::cout << "data object 2: " << dataObj2.get() << std::endl;
    }
    // ------
    {
         // Default template values <int, 0>
        Array defaultArr;
        // Empty string
        defaultArr.print();

        // stupidly just returns max
        int min = arrMin<int, 10>(defaultArr.data(), defaultArr.size());
        std::cout << min << std::endl;
    }
    // ------
    {
        // Nice Syntax with initializer_list
        // 1. Uniform Initialization        :   arr{1, 2, 3, 4, 5}
        //    Actually, we call constructor :   arr({1, 2, 3, 4, 5})
        // 2. Here we use Copy Operator
        // {x, y, z} in Stack
        // By default initializer_list don't implement deep copy, so (initializer_list={x, y}) is local only
        // Оказывается, std::vector при присваивании создает копию.
        // В имплементации приходится создавать копию, потому что initializer_list в Stack памяти.
        // Array<int, 5> arr = {1, 2, 3, 4};
        Array arr = {1, 2, 3, 4};
        arr.print();
        for(int i=0; i<arr.size(); i++){
            arr.set(i, i+1);
        }
        arr.print();
    }

}
