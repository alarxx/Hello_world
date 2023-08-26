package lambda_expression;

import java.util.function.Function;

/**
 * UnaryOperator<T> - T apply(T v)
 * BinaryOperator<T> - T apply(T v1, T v2)
 * Consumer<T> - void accept(T v)
 * Supplier<T> - T get()
 * Function<T, R> - R apply(T v)
 * Predicate<T> - boolean test(T v)
 * */
public class PredefinedInterfaces {
    public static void main(String[] args) {
        Function<Integer, Integer> factorial = (value) -> {
            int result = 1;
            for(int i=value; i>1; i--){
                result *= i;
            }
            return result;
        };

        System.out.println(factorial.apply(3));
    }
}
