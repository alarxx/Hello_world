package lambda_expression;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

interface MyF<T> {
    int func(T[] vals, T val);
}

class Comp {
    static <T> int countMatching(T[] vals, T val){
        int counter = 0;
        for(int i=0; i<vals.length; i++){
            if(vals[i] == val){
                counter++;
            }
        }
        return counter;
    }

    static <T extends Number> int compare(T v1, T v2){
        return (int) (v1.doubleValue() - v2.doubleValue());
    }
}

public class Pointer2GenMeth {
    static <T> int myOp(MyF<T> f, T[] vals, T val){
        return f.func(vals, val);
    }
    public static void main(String[] args) {
        Integer[] arr = new Integer[]{2, 2, 2, 1, 1, 3, 4, 5, 6, 7};
        System.out.println(myOp(Comp::<Integer>countMatching, arr, 2));


        List<Integer> list = new ArrayList<>();
        for(int i=0; i<10; i++)
            list.add(i);
        System.out.println(Collections.max(list, Comp::compare));
    }
}
