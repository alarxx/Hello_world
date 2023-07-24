package streams.files;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class Output {
    public static void main(String[] args) {
        // Copy file

        String fileName = "src\\streams\\files\\test.txt";
        String copyFileName = "src\\streams\\files\\copy_test.txt";

        FileInputStream fin = null;
        FileOutputStream fout = null;

        try{
            fin = new FileInputStream(fileName);
            fout = new FileOutputStream(copyFileName);
            while(true){
                int i = fin.read();
                if(i == -1){
                    break;
                }
                fout.write(i);
            }
        }
        catch(IOException e){
            e.printStackTrace();
        }
        finally{
            try{
                if(fin != null){
                    fin.close();
                }
            }
            catch(IOException e){
                e.printStackTrace();
            }

            try{
                fout.close();
            }
            catch(IOException e){
                e.printStackTrace();
            }
        }

    }
}
