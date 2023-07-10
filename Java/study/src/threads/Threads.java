package threads;

class MyRunnable implements Runnable {

    @Override
    public void run() {
        System.out.printf("%s started... \n", Thread.currentThread().getName());
        try{
            for(int i=0; i<5; i++){
                System.out.println(i);
                Thread.sleep(1000);
            }
        }
        catch(InterruptedException e){
            System.out.println("Thread has been interrupted");
        }
        System.out.printf("%s finished... \n", Thread.currentThread().getName());
    }

}

class MyThread extends Thread {
    public MyThread(){
        super("My Thread");
        System.out.println("Class Thread");
    }

    @Override
    public void run(){
        System.out.printf("%s started... \n", Thread.currentThread().getName());
        try{
            for(int i=0; i<5; i++){
                System.out.println(i);
                Thread.sleep(1000);
            }
        }
        catch(InterruptedException e){
            System.out.println("Thread has been interrupted");
        }
        System.out.printf("%s finished... \n", Thread.currentThread().getName());
    }
}

public class Threads {
    public static void main(String[] args) {
        Thread childThread = new Thread(new MyRunnable(), "Child Thread");
        System.out.println("childThread1: " + childThread);
        childThread.start();

        Thread childThread2 = new MyThread();
        System.out.println("childThread2: " + childThread2);
        childThread2.start();

        Thread t = Thread.currentThread();
        System.out.println("Current Thread:" + t);
        t.setName("My main Thread");
        System.out.println("Current Thread:" + t);

        try{
            for(int i=0; i<5; i++){
                System.out.println(i);
                t.sleep(1000);
            }
        }catch(Exception e){
            e.printStackTrace();
        }
    }
}
