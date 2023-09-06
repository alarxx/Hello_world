package sealed_classes;

public final class Alpha extends MySealedClass implements MySealedInterface {
    @Override
    public void interface_method() {
        System.out.println("Alpha interface_method");
    }
}
