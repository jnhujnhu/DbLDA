package Utils;

/**
 * Created by Kevin on 03/05/2017.
 */
public class ErrorHandler {
    public static void catchNaNError(int K, double[] paras, String parasName) {
        for(int i = 0; i < K; i ++)
            if(paras[i] != paras[i]) {
                System.out.format("NaN Error occur on parameters %s \n", parasName);
                for(int k = 0; k < K; k ++)
                    System.out.format("%.10f ", paras[k]);
                System.out.println();
                System.exit(0);
            }
    }
}
