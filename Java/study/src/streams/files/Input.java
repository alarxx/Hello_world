package streams.files;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class Input {
    public static void main(String[] args) {
        String fileName = "src\\streams\\files\\test.txt";

        FileInputStream fin;

        try{
            fin = new FileInputStream(fileName);
        }
        catch (FileNotFoundException e){
            System.out.println("Не удалось открыть файл.");
            e.printStackTrace();
            return;
        }

        try{
            int i;
            do{
                i = fin.read();
                if(i!=-1){
                    System.out.print((char) i);
                }
            }
            while(i != -1);
        }
        catch(IOException e){
            System.out.println("Ошибка при чтении файла.");
            e.printStackTrace();
        }

        try{
            fin.close();
            System.out.println("Файл закрыт.");
        }
        catch(IOException e){
            System.out.println("Не удалось закрыть файл.");
            e.printStackTrace();
        }
    }
}
