package enums;

enum Apple {
    Golden(12), Red(10), Green(9);
    private int price;
    Apple(int price){
        this.price = price;
    }
    Apple(){
        this.price = -1;
    }
    int getPrice(){
        return this.price;
    }
}

public class Enums {
    public static void main(String[] args) {
        Apple a = Apple.Golden;
        Apple b = Apple.Red;

        System.out.println(b.ordinal());

        if(a.equals(b)){
            System.out.println("a = b");
        }

        if(a == Apple.Red){
            System.out.println("if Apple is Red");
        } else {
            System.out.println("else if Apple is not Red");
        }

        switch(a){
            case Red:
                System.out.println("Apple is Red");
            case Golden:
                System.out.println("Apple is Golden");
            case Green:
                System.out.println("Apple is Green");
        }

        for(var variety: Apple.values()){
            System.out.println("price of " + variety + " apple is " + variety.getPrice());
        }
    }
}
