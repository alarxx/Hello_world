package threads;

class Queue {
    int n;
    boolean isValueSet = false;
    synchronized int get(){
        while(!isValueSet){
            try{
                System.out.println("Вызов на ожидание (get)");
                wait();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        System.out.println("get: " + n);
        isValueSet = false;
        notify();
        return n;
    }
    synchronized void set(int n){
        while(isValueSet){
            try{
                System.out.println("Вызов на ожидание (set)");
                wait();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        this.n = n;
        isValueSet = true;
        System.out.println("set: "+n);
        notify();
    }
}

class Producer implements Runnable{
    Queue q;
    Thread t;
    Producer(Queue q){
        this.q = q;
        t = new Thread(this, "Producer");

    }

    @Override
    public void run() {
        int i = 0;
        while(true){
            try {
                q.set(i++);
                Thread.sleep(500);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}

class Consumer implements Runnable {

    Queue q;
    Thread t;
    Consumer(Queue q){
        this.q = q;
        t = new Thread(this, "Consumer");
    }

    @Override
    public void run() {
        while(true){
            while(true){
                try {
                    q.get();
                    Thread.sleep(3000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}

public class WaitNotify {
    public static void main(String[] args) {
        // 2 потока пытаются взаимодействовать с 1-й очередью
        Queue q = new Queue();
        Producer p = new Producer(q);
        Consumer c = new Consumer(q);
        p.t.start();
        c.t.start();
    }
}
