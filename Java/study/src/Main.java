import java.util.Arrays;

class Solution {
    public int numJewelsInStones(String jewels, String stones) {
        int num = 0;
        for(int i=0; i<stones.length(); i++){
            int contains = jewels.indexOf(stones.charAt(i));
            if(contains != -1){
                num++;
            }
        }
        return num;
    }
}
public class Main {
    public static void main(String[] args) {
        String jewels = "z", stones = "ZZ";
        int num = new Solution().numJewelsInStones(jewels, stones);
        System.out.println(num);
    }

}
