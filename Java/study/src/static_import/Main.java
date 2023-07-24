package static_import;

import static java.lang.Math.*;
//import static java.lang.Math.pow;
//import static java.lang.Math.sqrt;

public class Main {
    public static void main(String[] args) {
        double a = 7, b = 11;
        // double hypot = Math.sqrt(Math.pow(a, 2) + Math.pow(b, 2));
        double hypot = sqrt(pow(a, 2) + pow(b, 2)); // Не нужно уточнять класс для использования статических методов Math
        System.out.println("a^2=" + pow(a, 2) + ", b^2=" + pow(b, 2) +", hypot=" + hypot);
    }
}
