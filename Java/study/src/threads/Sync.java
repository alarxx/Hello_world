package threads;

class Callme {
    /*synchronized*/ public void call(String msg){
        System.out.print("[" + msg);
        try{
            Thread.sleep(500);
        }catch(InterruptedException e){
            e.printStackTrace();
        }
        System.out.println("]");
    }
}

class Caller implements Runnable {
    String msg;
    Callme callme;
    Thread t;

    public Caller(Callme callme, String msg){
        this.msg = msg;
        this.callme = callme;
        this.t = new Thread(this);
    }

    @Override
    public void run(){
//        this.callme.call(this.msg);
        synchronized (this.callme){
            this.callme.call(this.msg);
        }
    }
}

public class Sync {

    public static void main(String[] args) {
        Callme callme = new Callme();
        Caller c1 = new Caller(callme, "Hello");
        Caller c2 = new Caller(callme, "Synchronized");
        Caller c3 = new Caller(callme, "World");
        c1.t.start();
        c2.t.start();
        c3.t.start();
        try{
            c1.t.join();
            c2.t.join();
            c3.t.join();
        }
        catch(InterruptedException e){
            System.out.println("Главный поток прерван");
        }
        System.out.println("Программа завершилась");
    }
}
