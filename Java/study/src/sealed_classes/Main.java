package sealed_classes;

// Невозможно наследоваться от final класса
//class AlphaChild extends Alpha {}

/**
 * Sealed class-ы должны давать permissions.
 * Классы которые получили permit должны быть sealed, non-sealed или final.
 * Все точно также и для интерфейсов.
 * */
public class Main {
    public static void main(String[] args) {

        MySealedClass alpha = new Alpha();
        Beta beta = new Beta();

        // alpha.interface_method();
        beta.interface_method();

        alpha.class_method();
        beta.class_method();

        BetaChild betaChild = new BetaChild();
        betaChild.interface_method();
        betaChild.class_method();
    }
}
