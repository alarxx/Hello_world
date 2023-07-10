package annotations;

import java.lang.annotation.Annotation;
import java.lang.annotation.Repeatable;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.reflect.Method;

@Retention(RetentionPolicy.RUNTIME)
@Repeatable(MyRepeatableAnnos.class)
@interface MyAnno1 {
    String str();
    int val() default 0;
}

@Retention(RetentionPolicy.RUNTIME)
@interface MyRepeatableAnnos { // Container for MyAnno
    MyAnno1[] value();
}

public class RepeatableAnnotations {

    @MyAnno1(str = "1")
    @MyAnno1(str = "2", val = 2)
    public static void myMeth(){
        System.out.println("kek");
    }

    public static void main(String[] args) {
        RepeatableAnnotations ob = new RepeatableAnnotations();

        try{
            Class<?> c = ob.getClass();
            Method m = c.getMethod("myMeth");
            System.out.println("\n1) Can get MyAnno1 by container annotation:");
            Annotation anno = m.getAnnotation(MyRepeatableAnnos.class);
            System.out.println(anno);

            System.out.println("\n2) Can get MyAnno1 by type:");
            for(Annotation a: m.getAnnotationsByType(MyAnno1.class)){
                System.out.println(a);
            }
        }
        catch(Exception e){
            e.printStackTrace();
        }

    }
}
