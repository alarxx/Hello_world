package mypack;

public class ExceptionCause {
    static void compute(int a) throws MyException {
        System.out.println("Вызов compute(" + a + ")");
        if(a > 10){
            throw new MyException(a);
        }
        System.out.println("Нормальное завершение");
    }

    static void demoproc() throws NullPointerException{
        NullPointerException e = new NullPointerException("Верхний уровень");
        e.initCause(new ArithmeticException("Причина")); // Сцепленные исключения, причина
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
        catch(NullPointerException | ArithmeticException e){ // Множественный перехват исключений
            System.out.println("Исключение перехвачено");
            System.out.println("Первоначальная причина " + e.getCause()); // Первоначальная причина исключения
        }
    }

}
