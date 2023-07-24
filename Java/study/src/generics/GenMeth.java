package generics;

public class GenMeth {
    static <T extends Comparable<T>, V extends T> boolean isIn(T x, V[] arr){
        for(int i=0; i<arr.length; i++){
            if(arr[i].equals(x)){
                return true;
            }
        }
        return false;
    }
    public static void main(String[] args) {
        Integer[] arr = new Integer[]{1, 2, 3, 4, 5};
        Integer x = 1;
        System.out.println(isIn(x, arr));

        String[] sarr = new String[]{"1", "2", "3", "4", "5"};
        String s = "11";
        System.out.println(isIn(s, sarr));
    }
}
