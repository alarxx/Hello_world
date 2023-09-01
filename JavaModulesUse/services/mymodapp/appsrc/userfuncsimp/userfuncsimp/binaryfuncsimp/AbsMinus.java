package userfuncsimp.binaryfuncsimp;

import userfuncs.binaryfuncs.*;

public class AbsMinus implements BinaryFunc {
	@Override
	public String getName(){
		return "absMinus";
	}
	@Override
	public int func(int a, int b){
		return Math.abs(a) - Math.abs(b);
	}
}