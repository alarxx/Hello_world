package lambda_expression;

interface MyFunc<T> {
    boolean func(T v1, T v2);
}

class HighTemp {
    private int hTemp;
    HighTemp(int hTemp){
        this.hTemp = hTemp;
    }
    boolean sameTemp(HighTemp ht2){
        return hTemp == ht2.hTemp;
    }
}

/**
 * Очень странно!
 * */
public class ObjectRefPointer {
    static <T> int conter(T[] vals, MyFunc<T> f, T v){
        int count = 0;
        for(int i=0; i<vals.length; i++){
            if(f.func(vals[i], v)){
                count++;
            }
        }
        return count;
    }

    public static void main(String[] args) {
        HighTemp[] weekDays = new HighTemp[10];
        for(int i=0; i<weekDays.length; i++){
            weekDays[i] = new HighTemp(i);
        }
        /* Объяснение из книги:
        * "Оба метода принимают параметр
        * типа HighTemp и возвращают булевский
        * результат. Следовательно, каждый метод
        * совместим с функциональным интерфейсом
        * MyFunc, поскольку тип вызывающего
        * объекта может быть сопоставлен с
        * первым параметром func(), а
        * аргумент - со вторым параметром func()"
        * */
        int count = conter(weekDays, HighTemp::sameTemp, new HighTemp(5));
        System.out.println(count);
    }
}
