package mypack;

public class Bank implements Callback {

	private interface NestedIFParent {
		default void whoami(){
			System.out.println("I am deafult interface method");
		}
		// Принадлежит только интерфейсу и больше никому
		static int defaultNumber(){ 
			return 0;
		}
		private int getFocus(){
			return -1;
		}
	}

	public interface NestedIF extends NestedIFParent {
		int NO = О;
		int YES = 1;
		// Можно реализовать этот вложенный интерфейс другим классом 
		// class C implements Bank.NestedIF
		boolean isNotNegative(int x); 
	}

	private int balance;

	public Bank(int balance){
		setBalance(balance);
	}

	@Override
	public void callback(int param){
		System.out.println("Banks callback realization " + param);
	}

	public int getBalance(){
		return this.balance;
	}

	public void setBalance(int balance){
		this.balance = balance;
	}
}