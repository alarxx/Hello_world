package switch_enhancements;

public class YieldListSwitch {
    public static void main(String[] args) {
        int eventCode = 6010;
        int priorityLevel = switch(eventCode){
            case 1000, 1205, 8900:
                yield 1;
            case 2000, 6010, 9128:
                yield 2;
            case 1002, 7023, 9300:
                yield 3;
            default:
                yield 0;
        };

        System.out.println("Priority level: " + priorityLevel);
    }
}
