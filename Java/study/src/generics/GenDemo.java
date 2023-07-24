package generics;

class NonGen {
    Object obj;
    NonGen(Object obj){
        this.obj = obj;
    }
    Object getObj(){
        return obj;
    }
    void showType(){
        System.out.println("Типом obj является: " + obj.getClass().getName());
    }
}

class Gen<T>{ // <T, V, E>
    T obj;
    Gen(T obj){
        this.obj = obj;
    }
    T getObj(){
        return obj;
    }
    void showType(){
        System.out.println("Типом T является " + obj.getClass().getName());
    }
}

class TwoGen<T, E> {

}

public class GenDemo {
    public static void main(String[] args) {
        System.out.println("\nInteger Gen");
        // Нельзя присвоить типа Double - обеспечение безопасности типов
        Gen<Integer> igen = new Gen<Integer>(123); // Присваивание, автоупаковка int->Integer
        igen.showType();
        int value = igen.getObj(); // распаковка Integer -> int
        System.out.println("Значение igen: " + value);

        System.out.println("\nString Gen");
        Gen<String> strgen = new Gen<>("Hello world");
        strgen.showType();
        String str = strgen.getObj();
        System.out.println("Значение strgen: " + str);

        // igen != strgen

        System.out.println("\nNonGen");
        NonGen strObj = new NonGen("Helol");
        strObj.showType();
        // String i = strObj.getObj();
        System.out.println("Значение strObj: " + strObj.getObj());

        NonGen iObj = new NonGen(123);
        iObj = strObj; // Действие концептуально неверно
        // int i = (int) iObj.getObj();
        System.out.println("Значение iObj: " + iObj.getObj());

        System.out.println("\nTwo Gen");
        TwoGen<Integer, String> twoGen = new TwoGen<>();
        System.out.println(twoGen);
    }
}
