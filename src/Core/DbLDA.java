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
    private int K, V, SliceLen;
    private Map<String, Integer> wordMap;
    private String dataPath;

    //Variational Parameters
    private int D, S, mD, mN, iter_No = 0;
    private double[][][] mu, sigma;
    private double[][] zeta;
    private double[][][][] gamma;
    private int[][] doc_word_slice;
    private int[][][] word;

    private double[][] mean_nkw;
    private double[] mean_nkd;
    private double convergeThreshold = 0.000001;

    public DbLDA(double alpha, double beta, double sigma, int K, int sliceLen) {
        this.Alpha = alpha;
        this.Beta = beta;
        this.Sigma = sigma;
        this.K = K;
        this.SliceLen = sliceLen;
        wordMap = new HashMap<>();
        V = 0;
    }

    public void initWordMap(String wordMapPath) throws FileNotFoundException {
        System.out.println("Init word map...");
        BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(new File(wordMapPath))));
        Scanner scanner = new Scanner(input);
        V = Integer.parseInt(scanner.next()); //word map No.
        while(scanner.hasNext()) {
            String[] temp = scanner.next().split(" ");
            wordMap.put(temp[0], Integer.parseInt(temp[1]));
        }
        scanner.close();
    }

    public void initParametersForVU(String DataDir) throws Exception {
        System.out.println("Init parameters...");
        if(V == 0)
            throw new Exception("Word map not initialized!");
        dataPath = DataDir;
        BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(new File(dataPath))));
        Scanner scanner = new Scanner(input);
        String[] temp = scanner.next().split(" ");

        S = Integer.parseInt(temp[0]); //slice count
        D = Integer.parseInt(temp[1]); //doc count
        mD = Integer.parseInt(temp[2]); //max doc per slice
        mN = Integer.parseInt(temp[3]); //max word per doc

        //init variational parameters
        mu = new double[S][mD][K];
        zeta = new double[S][mD];
        sigma = new double[S][mD][K];
        gamma = new double[S][mD][mN][K];

        for(int i = 0; i < S; i ++)
            for(int j = 0; j < mD; j ++) {
                zeta[i][j] = 0.1; //initial value of zeta? UNDER consideration.
                for (int k = 0; k < K; k ++) {
                    mu[i][j][k] = Math.log(Alpha)+ Math.log(mN/K);
                    sigma[i][j][k] = 0.1;
                    for(int n = 0; n < mN; n ++) {
                        gamma[i][j][n][k] = 1/K;
                    }
                }
            }


        word = new int[S][mD][mN];
        doc_word_slice = new int[S][mD + 1];
        mean_nkd = new double[K];
        mean_nkw = new double[K][V + 1];

        while(scanner.hasNext()) {
            String[] temp_doc = scanner.next().split(" ");
            int temp_slice = Integer.parseInt(temp_doc[0]) - 1;

            doc_word_slice[temp_slice][0]++;
            doc_word_slice[temp_slice][doc_word_slice[temp_slice][0]] = temp_doc.length - 1;

            for (int i = 1; i < temp_doc.length; i++) {
                word[temp_slice][doc_word_slice[temp_slice][0]][i - 1] = wordMap.get(temp_doc[i]);
                for(int j = 0; j < K; j ++) {
                    //init mean_nkw mean_nkd
                    mean_nkd[j] += gamma[temp_slice][doc_word_slice[temp_slice][0]][i - 1][j];
                    mean_nkw[j][word[temp_slice][doc_word_slice[temp_slice][0]][i - 1]] +=
                            gamma[temp_slice][doc_word_slice[temp_slice][0]][i - 1][j];
                }
            }
        }
        scanner.close();
    }

    private double mean_count_gamma(int ex_s, int ex_d, int ex_n, int k, int wsdn) {
        if(wsdn == 0)
            return mean_nkd[k] - gamma[ex_s][ex_d][ex_n][k];
        else
            return mean_nkw[k][wsdn] - gamma[ex_s][ex_d][ex_n][k];
    }

    public void iterateVariationalUpdate() throws FileNotFoundException {
        iter_No ++;
        for(int i = 0; i < S; i ++)
            for(int j = 0;j < doc_word_slice[S][0]; j ++) {
                double[] sum_gamma = new double[K];
                for (int n = 0; n < doc_word_slice[S][j + 1]; n++) {
                    //update gamma
                    double norm = 0;
                    double[] prev_gamma = new double[K];
                    for (int k = 0; k < K; k ++) {
                        prev_gamma[k] = gamma[i][j][n][k];
                        gamma[i][j][n][k] = (Beta + mean_count_gamma(i, j, n, k, word[i][j][n]))
                                / (V * Beta + mean_count_gamma(i, j, n, k, 0)) * Math.exp(mu[i][j][k]);
                        norm += gamma[i][j][n][k];
                    }
                    for (int k = 0; k < K; k ++) {
                        gamma[i][j][n][k] /= norm;
                        sum_gamma[k] += gamma[i][j][n][k];
                        //maintain mean_nkw mean_nkd
                        mean_nkw[k][word[i][j][n]] += gamma[i][j][n][k] - prev_gamma[k];
                        mean_nkd[k] += gamma[i][j][n][k] - prev_gamma[k];
                    }
                }
                //update mu
                for(int k = 0; k < K; k ++) {
                    mu[i][j][k] = Math.log(zeta[i][j] / doc_word_slice[i][j + 1] * ((Alpha - 1) + sum_gamma[k]))
                            - Math.pow(sigma[i][j][k], 2) /2;
                }

                //update sigma using Newton's method
                for(int k = 0; k < K; k ++) {
                    sigma[i][j][k] = NewtonsMethodforSigma(i, j, k, 0.1, 5);
                }

                //update zeta
                double temp_zeta = 0;
                for(int k = 0; k < K; k ++)
                    temp_zeta += Math.exp(mu[i][j][k] + Math.pow(sigma[i][j][k], 2) /2);
                zeta[i][j] = temp_zeta;
            }

    }

    private double NewtonsMethodforSigma(int s, int d, int k, double x0, int step) {
        double xp = x0, xa;
        for(int i = 0; i < step; i ++) {
            xa = xp - (xp * Math.exp(xp / 2) - zeta[s][d] / (4 * doc_word_slice[s][d + 1] * Math.exp(mu[s][d][k])))
            / ((1 + xp / 2) * Math.exp(xp / 2));
            xp = xa;
            if(xp < 0)
                System.out.println("Error occur when updating sigma!..");
        }
        return Math.sqrt(xp);
    }

    private double evaluateAscendency() {
        //TODO:compute difference after update
        return 0;
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
