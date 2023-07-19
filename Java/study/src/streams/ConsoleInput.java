package streams;

import java.io.*;

public class ConsoleInput {
    public static void main(String[] args){
        Console console = System.console();
        if(console == null) System.out.println("console is null");

        // Стандартный поток байтовых данных с клавиатуры
        InputStream inputStream = System.in;
        // Преобразователь байтовых данных в символьные
        InputStreamReader inputStreamReader = console == null ?
                new InputStreamReader(inputStream) :
                new InputStreamReader(inputStream, console.charset());
        // Буферизованный символьный поток
        BufferedReader bufferedReader = new BufferedReader(inputStreamReader);

        // ---
        System.out.println("Посимвольное чтение. \nВведите 'q' для завершения.");
        try {
            char c;
            do{
                c = (char) bufferedReader.read();
                System.out.println("char: " + c);
            }
            while(c != 'q');
        } catch (IOException e) {
            e.printStackTrace();
        }

        // ---
        System.out.println("Чтение по строкам. \nВведите 'stop' для завершения.");
        try {
            String str;
            do{
                str = bufferedReader.readLine();
                System.out.println("word: " + str);
            }
            while(!str.equals("stop"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        // ---
        System.out.println("Мини текстовый редактор. \nВведите 'stop' для завершения.");
        try{
            String[] strs = new String[100];
            for(int i=0; i<100; i++){
                strs[i] = bufferedReader.readLine();
                if(strs[i].equals("stop")){
                    break;
                }
            }
            for(int i=0; i<100; i++){
                if(strs[i].equals("stop")){
                    break;
                }
                System.out.println(i + ": " + strs[i]);
            }
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }
}
