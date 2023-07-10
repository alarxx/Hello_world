package annotations;

import java.lang.annotation.Annotation;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.reflect.Method;

@Retention(RetentionPolicy.RUNTIME)
@interface MyAnno {
    String str();
    int val();
}

@Retention(RetentionPolicy.RUNTIME)
@interface What {
    String description() default "Default. No description provided.";
}

// Маркерная аннотация
@Retention(RetentionPolicy.RUNTIME)
@interface MyMarker {}

@Retention(RetentionPolicy.RUNTIME)
@interface Single {
    String value();
    String second() default "Default.";
}

@Single("Some string value") // Сокращенная запись
@What(description = "Classes annotation")
@MyAnno(str = "Пример аннотации класса", val = 0)
public class Annotations {
    public static void main(String[] args) {
        myMeth(111);
    }

    @MyMarker
    @What // = @What() = @What(description = "Default")
    @MyAnno(str = "Пример аннотации", val = 123)
    public static void myMeth(int arg){

        Annotations ob = new Annotations();

        try{
            {
                System.out.println("Все аннотации класса:");
                for(Annotation annotation: ob.getClass().getAnnotations()){
                    System.out.println(annotation);
                }
                System.out.println();
            }

            // Получение одной аннотации с помощью рефлексии
            {
                Class<?> c = ob.getClass();
                Method m = c.getMethod("myMeth", int.class);
                MyAnno anno = m.getAnnotation(MyAnno.class); // литерал известного класса
                System.out.println("Аргументы аннотации MyAnno метода myMeth: " + anno.str() + " " + anno.val()); // получаем значение как методы

                System.out.println("myMeth has marker annotation: " + m.isAnnotationPresent(MyMarker.class));
            }

            {
                Class<?> c = ob.getClass();
                Method m = c.getMethod("myMeth", int.class);
                System.out.println("\nВсе аннотации метода:");
                for (Annotation annotation : m.getAnnotations()) {
                    System.out.println(annotation);
                }
                System.out.println();
            }
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

}
