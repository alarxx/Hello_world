package appstart.mymodappdemo;

import appfuncs.simplefuncs.SimpleMathFuncs;
import appsupport.supportfuncs.SupportFuncs;

public class MyModAppDemo {
	public static void main(String[] args){
		if(SupportFuncs.isFactor(2, 10)){
			System.out.println("2 is factor of 10");
		}

		System.out.println("lcd(35, 105) = " + SimpleMathFuncs.lcd(35, 105));

		System.out.println("gcd(35, 105) = " + SimpleMathFuncs.gcd(35, 105));
	}
}