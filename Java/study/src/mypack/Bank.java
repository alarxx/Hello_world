package mypack;

public class Bank {
    private int balance;

    public Bank(int balance){
        setBalance(balance);
    }

    public int getBalance(){
        return this.balance;
    }

    public void setBalance(int balance){
        this.balance = balance;
    }
}