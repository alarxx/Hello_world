package sealed_classes;

public class BetaChild extends Beta {
    @Override
    public void interface_method(){
        System.out.println("Beta's child interface_method");
    }

    @Override
    public void class_method(){
        System.out.println("Beta's child class_method");
    }
}
