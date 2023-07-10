package mypack;

public class Main {

    static void demoproc() throws NullPointerException {
        NullPointerException e = new NullPointerException("Верхний уровень");
        e.initCause(new ArithmeticException());
        throw e;
    }

    public static void main(String[] args){
        /*try{
            compute(1);
            compute(11);
        }
        catch(MyException e){
            System.out.println(e);
        }*/
        try{
            demoproc();
        }
        catch(NullPointerException e){
            System.out.println("Исключение перехвачено");
            System.out.println("Первоначальная причина " + e.getCause());
        }
    }

}
