package switch_enhancements;

public class ArrowSwitch {
    public static int getThree(){
        return 3;
    }
    public static void main(String[] args) {
        int labelled;

        int eventCode = 6010;

        int priorityLevel = switch(eventCode){
            case 1000, 1205, 8900 -> { // block
                System.out.println("yield 1");
                yield 1;
            }
            case 2000, 6010, 9128 -> labelled = 2;
            case 1002, 7023, 9300 -> getThree();
            default -> 0;
        };

        System.out.println("Priority level: " + priorityLevel);
    }
}
