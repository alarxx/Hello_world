package streams.files;

import java.io.FileInputStream;
import java.io.FileOutputStream;

/**
 * ARM - Automatic Resource Management.
 * Автоматическое управление ресурсами.
 * Автоматизация процесса закрытия потоков.
 * */
public class TryWithResources {
    public static void main(String[] args) {
        String fileName = "src\\streams\\files\\test.txt";
        String copyFileName = "src\\streams\\files\\copy_test.txt";
        try(
                var fin = new FileInputStream(fileName);
                var fout = new FileOutputStream(copyFileName)
        ){
            while(true){
                int i = fin.read();
                if(i == -1){
                    break;
                }
                fout.write(i);
            }
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }
}
