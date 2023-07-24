package generics;

class TwoD {
    int x, y;
    TwoD(int x, int y){
        this.x = x;
        this.y = y;
    }
}
class ThreeD extends TwoD {
    int z;
    ThreeD(int x, int y, int z){
        super(x, y);
        this.z = z;
    }
}
class FourD extends ThreeD {
    int t;
    FourD(int x, int y, int z, int t){
        super(x, y, z);
        this.t = t;
    }
}

class Coords<T extends TwoD>{
    T[] coords;
    Coords(T[] coords){
        this.coords = coords;
    }
}

public class BoundedWildcard {
    static void showXY(Coords<?> coords){
        System.out.println("showXY");
        for(int i=0; i<coords.coords.length; i++){
            System.out.println("x: " + coords.coords[i].x + ", y: " + coords.coords[i].y);
        }
    }
    /**
     * Можем использовать только производные от ThreeD классы.
     * // Так же можно использовать <? super subclass>,
     * // то есть только родительные классы subclass-а
     * */
    static void showXYZ(Coords<? extends ThreeD> coords){
        System.out.println("showXYZ");
        for(int i=0; i<coords.coords.length; i++){
            System.out.println("x: " + coords.coords[i].x + ", y: " + coords.coords[i].y + ", z: " + coords.coords[i].z);
        }
    }
    public static void main(String[] args) {
        System.out.println("TwoD");
        TwoD[] arrayTwoD = new TwoD[]{
                new TwoD(21, 21),
                new TwoD(22, 22)
        };
        Coords<TwoD> coordsTwoD = new Coords<>(arrayTwoD);
        showXY(coordsTwoD);
//        showXYZ(coordsTwoD);

        System.out.println("ThreeD");
        ThreeD[] arrayThreeD = new ThreeD[]{
                new ThreeD(31, 31, 31),
                new ThreeD(32, 32, 32)
        };
        Coords<ThreeD> coordsThreeD = new Coords<>(arrayThreeD);
        showXY(coordsThreeD);
        showXYZ(coordsThreeD);
    }
}
