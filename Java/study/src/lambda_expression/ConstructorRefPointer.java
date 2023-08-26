package lambda_expression;

import java.lang.reflect.Method;

interface I<R, T>{
    // Метод - Конструктор.
    // Возвращает объект класса конструктора.
    R func(T t);
}

class SimpleClass<T> {
    private T t;
    SimpleClass(T t){
        this.t = t;
    }
    public T getT(){
        return t;
    }
}

public class ConstructorRefPointer {
    /**
     * метод получает метод(конструктор) определенного типа,
     * подходящего под метод интерфейса, и расскрывает подробности класса.
     * */
    static <R, T> void details(I<R, T> i, T t){
        R ob = i.func(t);

        Class<?> c = ob.getClass();
        System.out.println(c.getName());
        Method[] methods = c.getDeclaredMethods();
        for(Method m: methods){
            System.out.println(m.getName());
        }
    }
    public static void main(String[] args) {
        // Ссылка на конструктор
        I<SimpleClass<Double>, Double> i = SimpleClass<Double>::new;
        SimpleClass<Double> simpleClass = i.func(1.05d);
        System.out.println(simpleClass.getT());

        details(i, 1d);
    }
}
