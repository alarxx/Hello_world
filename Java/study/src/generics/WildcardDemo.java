package generics;

/**
 *  Bounds. Bounded with Number.
 * */
class Stats<T extends Number>{
    T[] nums;
    Stats(T[] nums){
        this.nums = nums;
    }
    double average(){
        double sum = 0d;
        for(int i=0; i<nums.length; i++){
            sum += nums[i].doubleValue();
        }
        return sum / nums.length;
    }
    /**
     * <?> - wildcard.
     * */
    boolean isSameAverage(Stats<?> stats){
        return average() == stats.average();
    }
}

public class WildcardDemo {
    public static void main(String[] args) {
        Integer[] inums = new Integer[]{1, 2, 3};
        Stats<Integer> iOb = new Stats<>(inums);


//        Double[] dnums = new Double[]{1d, 2d, 3d};
        Double[] dnums = new Double[]{1.1d, 2d, 3d};
        Stats<Double> dOb = new Stats<>(dnums);

        if(iOb.isSameAverage(dOb)){
            System.out.println("Одинаковые средние.");
        }
        else {
            System.out.println("Разные средние.");
        }
    }
}
