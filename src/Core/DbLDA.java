package Core;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

/**
 * Created by Kevin on 26/04/2017.
 */
public class DbLDA {

    //Model Parameters
    private double Alpha, Beta, Sigma;
    private int K, wordMapCount, SliceLen;
    private Map<String, Integer> wordMap;
    private String dataPath;

    //Variational Parameters
    private int D, S, mD, mN, iter_No = 0;
    private double[][][] mu, sigma;
    private double[][] zeta;
    private double[][][][] gamma;

    private double convergeThreshold = 0.000001;

    public DbLDA(double alpha, double beta, double sigma, int K, int sliceLen) {
        this.Alpha = alpha;
        this.Beta = beta;
        this.Sigma = sigma;
        this.K = K;
        this.SliceLen = sliceLen;
        wordMap = new HashMap<>();
    }

    public void initWordMap(String wordMapPath) throws FileNotFoundException {
        BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(new File(wordMapPath))));
        Scanner scanner = new Scanner(input);
        wordMapCount = Integer.parseInt(scanner.next()); //word map No.
        while(scanner.hasNext()) {
            String[] temp = scanner.next().split(" ");
            wordMap.put(temp[0], Integer.parseInt(temp[1]));
        }
    }

    public void initParametersForVU(String DataDir) throws FileNotFoundException {
        dataPath = DataDir;
        BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(new File(dataPath))));
        Scanner scanner = new Scanner(input);
        String[] temp = scanner.next().split(" ");
        scanner.close();
        S = Integer.parseInt(temp[0]); //slice count
        D = Integer.parseInt(temp[1]); //doc count
        mD = Integer.parseInt(temp[2]); //max doc per slice
        mN = Integer.parseInt(temp[3]); //max word per doc

        mu = new double[S][mD][K];
        zeta = new double[S][mD];
        sigma = new double[S][mD][K];
        gamma = new double[S][mD][mN][K];

        for(int i = 0; i < S; i ++)
            for(int j = 0; j < mD; j ++) {
                zeta[i][j] = 0.1; //initial value of zeta? UNDER consideration.
                for (int k = 0; k < K; k++) {
                    mu[i][j][k] = Math.log(Alpha)+ Math.log(mN/K);
                    sigma[i][j][k] = 0.1;
                    for(int n = 0; n < mN; n ++) {
                        gamma[i][j][n][k] = 1/K;
                    }
                }
            }
    }

    private double mean_count_gamma() {
        return 0.0;
    }

    public void iterateVariationalUpdate() throws FileNotFoundException {
        iter_No ++;
        BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(new File(dataPath))));
        Scanner scanner = new Scanner(input);
        scanner.next(); //ignore first line

        while(scanner.hasNext()) {

        }
    }

    public int getCurrentIterateCount() {
        return iter_No;
    }

    public void setAlpha(double a) {
        Alpha = a;
    }

    public void setBeta(double b) {
        Beta = b;
    }

    public void setSigma(double s) {
        Sigma = s;
    }

    public void setK(int k) {
        K = k;
    }
}
