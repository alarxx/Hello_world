package appstart.mymodappdemo;

import java.util.ServiceLoader;

import appfuncs.simplefuncs.SimpleMathFuncs;
import userfuncs.binaryfuncs.*;

public class MyModAppDemo {
	public static void main(String[] args){
		if(SimpleMathFuncs.isFactor(2, 10)){
			System.out.println("2 is factor of 10");
		}

		System.out.println("lcd(35, 105) = " + SimpleMathFuncs.lcd(35, 105));

		System.out.println("gcd(35, 105) = " + SimpleMathFuncs.gcd(35, 105));


		ServiceLoader<BinaryFuncProvider> ldr = ServiceLoader.load(BinaryFuncProvider.class);

		for(BinaryFuncProvider bfp : ldr){
			System.out.println("---------");
			BinaryFunc binaryFunc = bfp.get();
			System.out.println(binaryFunc.getName());
			int a = 3, b = -4;
			System.out.println("func(" + a + ", " + b + ") = " + binaryFunc.func(a, b));
		}
	}
}