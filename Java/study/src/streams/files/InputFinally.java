package streams.files;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class InputFinally {
    public static void main(String[] args) {
        FileInputStream fin;

        try{
            fin = new FileInputStream("src\\streams\\files\\test.txt");
        }
        catch(FileNotFoundException e){
            System.out.println("Не удалось открыть файл. " + e);
            return;
        }

        try{
            while(true){
                int i = fin.read();
                if(i == -1){
                    break;
                }
                System.out.print((char) i);
            }
            // fin.close() - вот так делать нежелательно
        }
        catch(Exception e){
            System.out.println("Ошибка при чтении файла. " + e);
        }
        finally {
            // Файл будет закрыт в любом случае
            try{
                fin.close();
                System.out.println("Файл закрыт.");
            }
            catch(IOException e){
                e.printStackTrace();
            }
        }
    }
}
