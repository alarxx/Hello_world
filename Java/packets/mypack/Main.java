package mypack;

import external.Personal;

public class Main{
	public static void main(String[] args){
		Bank bank = new Bank(123);
		System.out.println(bank.getBalance());

		Callback c = bank;
		c.callback(100);

		Personal p = new Personal();
	}
}