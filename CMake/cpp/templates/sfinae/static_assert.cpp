#include <utility>

using std::void_t;

template< class , class = void >
struct has_member : std::false_type
{ };

// specialized as has_member< T , void > or discarded (SFINAE)
template< class T >
struct has_member< T , void_t< decltype( T::member ) > > : std::true_type
{ };

class A {
public:
    int member;
};

class B {
};

int main(){
    static_assert( has_member< A >::value , "A" );
    static_assert( has_member< B >::value , "B" );
}
