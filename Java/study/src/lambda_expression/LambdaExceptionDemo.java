package lambda_expression;

class EmptyArrayException extends Exception {
    public EmptyArrayException(){
        super("Empty array");
    }
}

interface DoubleArrayI {
    double func(double[] arr) throws EmptyArrayException;
}

public class LambdaExceptionDemo {
    public static void main(String[] args) throws EmptyArrayException {
        DoubleArrayI mean = (arr)->{
          if(arr.length == 0){
              throw new EmptyArrayException();
          }
          double sum = 0;
          for(int i=0; i<arr.length; i++){
              sum += arr[i];
          }
          return sum / arr.length;
        };

        System.out.println(mean.func(new double[]{1d, 2d, 3d}));
        System.out.println(mean.func(new double[]{}));
        System.out.println("Не выполнится");
    }
}
