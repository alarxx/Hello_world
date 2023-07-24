package assert_example;

/**
 * Для запуска с проверкой утверждения нужно добавить флаг -ea
 * java -ea Main.java
 * */
class Main {
	public static int number = 4;
	public static int getnum(){
		return number--;
	}
	public static void main(String[] args){
		for(int i=0; i<100; i++){
			int n = getnum();
			assert n > 0: "My Error!!!";
			System.out.println(n);			
		}
	}
}