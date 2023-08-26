package lambda_expression;

interface MyInterface {
    int f(int v);
}

class MyClass {
    private int privateField;
    MyClass(int privateField){
        this.privateField = privateField;
    }
    private static int intOp(MyInterface myInterface, int v){
        return myInterface.f(v);
    }
    void myMethod(){
        int cant = 1; // effectively final variable

        System.out.println("result of intOp = " + intOp((v)->{
            this.privateField *= 2;

            System.out.println("The cant = " + cant);
//            cant *= 2;

            return this.privateField;
        }, this.privateField));

//        cant *= 3;

        System.out.println("this.privateField now = " + this.privateField);
    }
}

/**
 * Лямбда-выражения и захват переменных
 * */
public class LambdaVariableCapture {
    public static void main(String[] args) {
        MyClass myClass = new MyClass(123);
        myClass.myMethod();
    }
}
