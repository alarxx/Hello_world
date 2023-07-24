package generics;



public class Hierarchy {
    public static void main(String[] args) {new Hierarchy().main();}

    class Gen<T> {
        protected T ob;
        Gen(T ob){
            setOb(ob);
        }
        public T getOb() {
            System.out.println("Gen");
            return ob;
        }

        public void setOb(T ob) {
            this.ob = ob;
        }
    }
    class Gen2<T> extends Gen<T>{
        Gen2(T ob){
            super(ob);
        }
        @Override
        public T getOb(){
            System.out.println("Gen2");
            return ob;
        }
    }

    void main(){
        Gen<Integer> gen = new Gen<>(123);
        System.out.println(gen.getOb());
        System.out.println(gen instanceof Gen<?>);
        System.out.println(gen instanceof Gen2<?>);

        Gen2<Integer> gen2 = new Gen2<>(124);
        System.out.println(gen2.getOb());
        System.out.println(gen2 instanceof Gen<?>);
        System.out.println(gen2 instanceof Gen2<?>);

        Gen<Integer> genConverted = (Gen<Integer>) gen2;
        System.out.println(genConverted.getOb());

//        Gen<Integer>[] arr = new Gen<Integer>[10];
        Gen<?>[] arr = new Gen<?>[10];
    }
}
