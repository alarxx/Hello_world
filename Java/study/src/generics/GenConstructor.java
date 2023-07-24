package generics;

class GenCons {
    private double val;

    <T extends Number> GenCons(T val){
        setVal(val.doubleValue());
    }

    public double getVal() {
        return val;
    }

    public void setVal(double val) {
        this.val = val;
    }
}
public class GenConstructor {
    public static void main(String[] args) {
        GenCons genCons = new GenCons(123);
        System.out.println(genCons.getVal());
    }
}
