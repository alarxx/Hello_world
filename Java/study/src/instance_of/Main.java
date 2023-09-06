package instance_of;

public class Main {
    public static void main(String[] args) {
        Number number = Integer.valueOf(123);

        System.out.print("number instanceof Integer ");
        System.out.println(number instanceof Integer);

        System.out.print("number instanceof Double ");
        System.out.println(number instanceof Double);
        System.out.println();


        // Pattern matching version of instanceof
        if(number instanceof Double dObj) {
            System.out.println("dObj refers to a double: " + dObj);
        }
        else if(number instanceof Integer iObj){
            System.out.println("iObj refers to an integer: " + iObj);
        }

        if(number instanceof Integer iObj && iObj >= 0){
            System.out.println("iObj is greater than zero");
        }

        // Illegal
        // if(number instanceof Integer iObj & iObj >= 0) {}
        System.out.println();


        Object[] someObjects = {
                new String("Alpha"),
                new String("Beta"),
                new String("Gamma"),
                Integer.valueOf(123)
        };

        // for loop
        for(int i = 0; someObjects[i] instanceof String str && i < someObjects.length; i++){
            System.out.println(str);
        }

        // Same but with while loop
        int i = 0;
        while(someObjects[i] instanceof String str && i < someObjects.length){
            System.out.println("Processing: " + str);
            i++;
        }

        // Illegal
       /* do{
            System.out.println("Processing: " + str);
            i++;
        }
        while(someObjects[i] instanceof String str && i < someObjects.length);*/
    }
}
