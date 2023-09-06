package sealed_classes;

public sealed class MySealedClass permits Alpha, Beta {
    public void class_method(){
        System.out.println("MySealedClass method");
    }
}
