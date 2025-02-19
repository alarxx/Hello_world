#include <iostream>
#include <memory>

// --- Data ---
class Data {
private:
    int __theData;
public:
    Data(int data) : __theData(data) {
        std::cout << "Data Constructor" << "\n";
    }
    ~Data() {
        std::cout << "~Data Destructor" << "\n";
    }
    int get(){ return __theData; }
};
// ------

template <typename T> void fun(std::weak_ptr<T> wptr){
    // Перед функцией создасться копия weak_ptr-а, который мы передаем
    // wptr - это копия
    std::cout << "---start-funny fun function---" << "\n";
    if(std::shared_ptr<Data> tmp = wptr.lock()){
        std::cout << "use count: " << tmp.use_count() << std::endl;
    }
    else {
        std::cout << "wptr is expired!" << std::endl;
    }
    std::cout << "---end--funny fun function---" << "\n\n";
}

// By default мы должны работать вот так! call-by-value
template <typename T> void sayHello(std::shared_ptr<T> sptr){
    // Перед функцией создасться копия shared_ptr-а, который мы передаем
    // sptr - это копия
    std::cout << "(sayHello) Call by value" << std::endl;
    std::cout << "(sayHello) use count: " << sptr.use_count() << std::endl; // 2
}
// Мы никогда не должны создавать pointer to SmartPointer.
template <typename T> void sayHello(std::shared_ptr<T> * sptr){
    std::cout << "(sayHello) Call by pointer" << std::endl;
    std::cout << "(sayHello) use count: " << sptr->use_count() << std::endl;
}


int main(){
    // MySmartPointer<Data> sptr(new Data(123));
    std::shared_ptr<Data> sptr = std::make_shared<Data>(123);
    // *sptr - Data object

    std::cout << "1. use count: " << sptr.use_count() << std::endl; // 1

    // Создаёт копию
    sayHello(sptr);
    // Удалит копию

    {
        std::cout << "{}" << std::endl;

        std::shared_ptr<Data> sptr2 = sptr;
        std::cout << "2. use count: " << sptr.use_count() << std::endl; // 2
    } // sptr2 deleted

    std::cout << "3. use count: " << sptr.use_count() << std::endl;

    // Как отследить, что sptr удалился?
    // Мы не можем просто создать другой shared_ptr, чтобы отслеживать sptr
    // std::shared_ptr<Data> sptr2 = sptr; // No
    // Circular Dependency of shared pointers never deallocates what leads to Memory Leak
    std::weak_ptr<Data> weak1 = sptr;
    // При передаче создасться копия weak_ptr
    // внутри мы будем вызывать lock() -> shared_ptr
    fun(weak1);

    std::cout << "---reset---" << std::endl;
    sptr.reset(new Data(321));
    // Теперь он ссылается на новый ControlBlock
    // И соответсвенно уменьшил владельцев старого ControlBlock-а
    // Но так же обычно не делают, меняют значение на которое указывает указатель, а не полностью объект
    std::cout << "4. use count: " << sptr.use_count() << std::endl;

    // weak1 is expired!
    // expired() лучше не использовать и предпочесть lock() из-за race condition
    // weak1 все еще ссылается на старый ControlBlock
    // старый ControlBlock не удаляется, так как все ещё имеет наблюдающих-weak_ptr
    // Не удаляем, чтобы оставшиеся weak_ptr не указывали на мусор
    if(std::shared_ptr<Data> tmp = weak1.lock()){
        // tmp создается из старого ControlBlock
        // tmp's controlBlock.object is nullptr, поэтому bool(tmp) false
        std::cout << "5. use count: " << tmp.use_count() << std::endl;
    }
    else {
        std::cout << "weak1 is expired!" << std::endl;
    }
}

