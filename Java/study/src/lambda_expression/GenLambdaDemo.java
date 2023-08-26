package lambda_expression;

interface SomeFunc<T> {
    T func(T t);
}

public class GenLambdaDemo {
    // Вот так, если что, создаются обобщенные методы
    // public static <T> strOp(SomeFunc<T> someFunc, String str){}
    public static String strOp(SomeFunc<String> someFunc, String str){
        return someFunc.func(str);
    }

    public static void main(String[] args) {

        SomeFunc<String> reverse = (var str)->{
            String result = "";
            for(int i=str.length()-1; i>=0; i--){
                result += str.charAt(i);
            }
            return result;
        };

        System.out.println(strOp(reverse, "Alar"));

    }
}
