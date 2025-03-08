#include <iostream>
#include <vector>
#include <algorithm>

void unary_example(){
    int arr[]{1, 2, 3, 4, 5, 6};
    int size = sizeof(arr) / sizeof(arr[0]);

    int res[size];

    // res = map(arr, ()=>{})
    std::transform(arr, arr + size, res, [](int a){
        return a * 2;
    });

    for(auto i: res){
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

void binary_example(){
    std::vector<int> arr = {1, 2, 3, 4, 5, 6};

    std::vector<int> arr2(arr.size());
    {
        int i = 1;
        for(std::vector<int>::iterator it = arr2.begin(); it != arr2.end(); it++, i++){
            *it = i;
        }
    }

    std::vector<int> res(arr.size());

    std::transform(arr.begin(), arr.end(), arr2.begin(), res.begin(), [](int a, int b){
        return a * b;
    });

    for(auto i: res){
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

int main(){
    // unary_example();
    binary_example();
}
