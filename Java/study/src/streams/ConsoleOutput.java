package streams;

import java.io.OutputStream;
import java.io.PrintWriter;

public class ConsoleOutput {
    public static void main(String[] args){
        // Поток байтовых данных
        int b = 'A';
        System.out.write(b);
        System.out.write('\n');
        System.out.write('\n');

        // Поток символьных данных. Упрощает интернационализацию программ.
        OutputStream outputStream = System.out;
        boolean flashingOn = true;
        PrintWriter pw = new PrintWriter(outputStream, flashingOn);
        pw.println("Hello");
        pw.println(-7);
        pw.println(4.5e-7);
    }
}
