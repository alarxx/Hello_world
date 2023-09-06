package records;

record Employee(String name, int idNum){
    static int pendingId = -1;

    /*
    // Canonical constructor
    Employee(String name, int idNum){
        this.name = name.trim();
        this.idNum = idNum;
    }*/

    // Canonical constructor
    // Compact record constructor
    Employee {
        name = name.trim();
        int     i = name.indexOf(','),
                j = name.lastIndexOf(',');
        if(i != j){
            throw new IllegalArgumentException("Multiple commas found.");
        }
        // Одновременная проверка
        // Если запятой нет или если имя без фамилии "name,"
        if(i < 1 | name.length() == i + 1){
            throw new IllegalArgumentException("Required format: last, first");
        }
    }

    // Non-canonical constructor
    Employee(String name){
        this(name, pendingId);
    }

    String firstName(){
        return name.substring(0, name.indexOf(','));
    }
}
public class Main {
    public static void main(String[] args) {
        Employee alar = new Employee("Alar, Akilbekov", 100);
        Employee ulan = new Employee("Ulan, Obezyana");


        System.out.println(alar.firstName());
        System.out.println(alar.idNum());

        System.out.println(ulan.firstName());
        System.out.println(ulan.idNum());

        // ошибка
        Employee amir = new Employee("Bomber");
    }
}
