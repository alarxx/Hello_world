package lambda_expression;

interface StringFunc {
    String func(String s);
}

class StringOps {
    static String reverse(String s){
        String result = "";
        for(int i=s.length()-1; i>=0; i--){
            result += s.charAt(i);
        }
        return result;
    }

    String doubled(String s){
        return s + s;
    }
}

public class MethodPointer {
    static String strOp(StringFunc stringFunc, String s){
        return stringFunc.func(s);
    }

    public static void main(String[] args) {
        System.out.println(strOp(StringOps :: reverse, "Hello"));

        StringOps stringOps = new StringOps();
        System.out.println(strOp(stringOps :: doubled, "Alar"));
    }
}
