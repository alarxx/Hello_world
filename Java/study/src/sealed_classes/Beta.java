package sealed_classes;

public non-sealed class Beta extends MySealedClass implements MySealedInterface {
    @Override
    public void interface_method() {
        System.out.println("Beta interface_method");
    }
}
