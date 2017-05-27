package Core;

import Utils.ErrorHandler;
import Utils.Sampler;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

/**
 * Created by Kevin on 26/04/2017.
 */
public class DbLDA {

    //Model Parameters
    private double Alpha, Beta, Sigma;
    private int K, V, SliceLen;
    private Map<String, Integer> wordMap;
    private String outputDir;

    //Variational Parameters
    private double[][][] mu, sigma;
    private double[][] zeta;
    private double[][][][] gamma;
    private double[][] mean_nkw; //variational mean of word count with topic k and word w
    private double[] mean_nkd;//variational mean of word count with topic k

    //D: documents count,  S:slice count,  mD: max number of doc in one slice,  mN: max number of word in one doc
    private int D, S, mD, mN, iter_No = 0;

    //doc_word_slice[s][0]: number of doc in slice s
    //doc_word_slice[s][1-Ds]:number of word in doc(s,d)
    private int[][] doc_word_slice;

    private int[][][] word;
    private double[][] Phi = null;

    //flag of whether to store phi after each iteration
    private boolean flag_Phi = false;

    /**
     * DbLDA constructor
     * @param alpha
     * @param beta
     * @param sigma                     :deviation inside one subset
     * @param K                         :number of topic
     * @param sliceLen                  :number of slice in one subset
     * @param outputDir                 :output(Phi, Theta) directory
     * @param storePhiAfterEveryIter    :whether to store phi after each iteration
     */
    public DbLDA(double alpha, double beta, double sigma, int K, int sliceLen, String outputDir
            , boolean storePhiAfterEveryIter) {
        this.Alpha = alpha;
        this.Beta = beta;
        this.Sigma = sigma;
        this.K = K;
        this.SliceLen = sliceLen;
        this.outputDir = outputDir;
        wordMap = new HashMap<>();
        V = 0;
        flag_Phi = storePhiAfterEveryIter;
    }

    /**
     * Initialize word map
     * @param wordMapPath
     * @throws FileNotFoundException
     */
    public void initWordMap(String wordMapPath) throws FileNotFoundException {
        System.out.println("Init word map...");
        BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(new File(wordMapPath))));
        Scanner scanner = new Scanner(input);
        V = Integer.parseInt(scanner.nextLine()); //word map No.
        while (scanner.hasNextLine()) {
            String[] temp = scanner.nextLine().split(" ");
            wordMap.put(temp[0], Integer.parseInt(temp[1]));
        }
        scanner.close();
    }

    /**
     * Fix Phi(topic-word) when iterating(Estimate Topic Proportion for an Unseen Document)
     * @param phiPath
     * @throws Exception
     */
    public void givenPhi(String phiPath) throws Exception {
        Phi = new double[K][V];
        Perplexity.readFromFile(Phi, phiPath);
    }

    /**
     * Random Initialize all Parameters
     * @param DataDir                   :corpus directory
     * @param first_timeslice           :first slice label(1-27:1/28-31:28)
     * @param doc_percent               :using how much percent of word in one document
     * @throws Exception
     */
    public void initParametersForVU(String DataDir, int first_timeslice, double doc_percent) throws Exception {
        System.out.println("Init parameters...");
        if(V == 0)
            throw new Exception("Word map not initialized!");
        BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(new File(DataDir))));
        Scanner scanner = new Scanner(input);
        String[] temp = scanner.nextLine().split(" ");

        S = (int) Math.ceil((double) Integer.parseInt(temp[0]) / (double) SliceLen); //slice count
        D = Integer.parseInt(temp[1]); //doc count
        mD = Integer.parseInt(temp[2]) * SliceLen; //max doc per slice
        mN = (int) Math.floor(Integer.parseInt(temp[3]) * doc_percent); //max word per doc

        //init variational parameters.
        mu = new double[S][][];
        zeta = new double[S][];
        sigma = new double[S][][];  //sigma-----deviation(NOT standard deviation)
        gamma = new double[S][][][];

        word = new int[S][mD][];
        doc_word_slice = new int[S][mD + 1];
        mean_nkd = new double[K];
        mean_nkw = new double[K][V];

        int slice_no = 0, temp_slice_begin = 0;
        //init doc matrix
        while(scanner.hasNextLine()) {
            String[] temp_doc = scanner.nextLine().split(" ");
            int temp_slice = Integer.parseInt(temp_doc[0]) - first_timeslice;
            int doc_length = (int) Math.floor((temp_doc.length - 1) * doc_percent);

            if(temp_slice - temp_slice_begin == SliceLen ) {
                slice_no ++;
                temp_slice_begin = temp_slice;
            }
            //doc_word_slice[s][0]:Ds  doc_word_slice[s][d + 1]:Nsd
            doc_word_slice[slice_no][0] ++;
            doc_word_slice[slice_no][doc_word_slice[slice_no][0]] = doc_length;
            word[slice_no][doc_word_slice[slice_no][0] - 1] = new int[doc_length];

            for (int i = 0; i < doc_length; i++) {
                word[slice_no][doc_word_slice[slice_no][0] - 1][i] = wordMap.get(temp_doc[i + 1]);
            }

        }
        scanner.close();

        //init matrix
        for(int s = 0; s < S; s ++) {
            mu[s] = new double[doc_word_slice[s][0]][K];
            zeta[s] = new double[doc_word_slice[s][0]];
            sigma[s] = new double[doc_word_slice[s][0]][K];
            gamma[s] = new double[doc_word_slice[s][0]][][];
            for(int d = 0; d < doc_word_slice[s][0]; d ++) {
                gamma[s][d] = new double[doc_word_slice[s][d + 1]][K];
            }
        }

        //random initialize parameters
        for(int s = 0; s < S; s ++) {
            double[] theta;
            theta = Sampler.getDirichletSample(K, Alpha);
            for(int k0 = 0; k0 < K; k0 ++)
                theta[k0] = Math.log(theta[k0]);
            for(int d = 0; d < doc_word_slice[s][0]; d ++) {
                //init zeta
                zeta[s][d] = new Random().nextDouble() * 0.1 + 0.1001;
                double[] b_sigma = new double[K];
                for(int k2 = 0; k2 < K; k2 ++) {
                    b_sigma[k2] = Sigma;
                }
                //init mu
                mu[s][d] = Sampler.getGaussianSample(K, theta, b_sigma);

                //init sigma
                for(int k1 = 0; k1 < K; k1 ++)
                    sigma[s][d][k1] = new Random().nextDouble() * 0.4 + 0.5001;
                for(int n = 0; n < doc_word_slice[s][d + 1]; n ++) {
                    //init gamma
                    gamma[s][d][n] = Sampler.getGaussianSample(K, mu[s][d], sigma[s][d]);

                    double gamma_norm = 0;
                    for(int k = 0; k < K; k ++) {
                        gamma_norm += Math.exp(gamma[s][d][n][k]);
                    }

                    //init mean_nkw mean_nkd
                    for(int k = 0; k < K; k ++) {
                        gamma[s][d][n][k] = Math.exp(gamma[s][d][n][k]) / gamma_norm;
                        mean_nkd[k] += gamma[s][d][n][k];
                        mean_nkw[k][word[s][d][n] - 1] += gamma[s][d][n][k];
                    }
                }
            }
        }
    }

    /**
     * Get Variational Mean of Count(k,d) or Count(k)
     * @param ex_s
     * @param ex_d
     * @param ex_n
     * @param k
     * @param wsdn
     * @return
     */
    private double mean_count_gamma(int ex_s, int ex_d, int ex_n, int k, int wsdn) {
        if(wsdn == 0)
            return mean_nkd[k] - gamma[ex_s][ex_d][ex_n][k];
        else
            return mean_nkw[k][wsdn - 1] - gamma[ex_s][ex_d][ex_n][k];
    }

    /**
     * Execute One Iteration
     * @return
     * @throws Exception
     */
    public boolean iterateVariationalUpdate() throws Exception {
        iter_No++;
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < doc_word_slice[i][0]; j++) {
                double[] sum_gamma = new double[K];
                for (int n = 0; n < doc_word_slice[i][j + 1]; n++) {
                    //update gamma
                    double norm = 0;
                    double[] prev_gamma = new double[K];

                    for (int k = 0; k < K; k++) {
                        prev_gamma[k] = gamma[i][j][n][k];
                        if (Phi == null) {
                            gamma[i][j][n][k] = (Beta + mean_count_gamma(i, j, n, k, word[i][j][n])) * Math.exp(mu[i][j][k])
                                    / (V * Beta + mean_count_gamma(i, j, n, k, 0));
                        } else {
                            gamma[i][j][n][k] = Phi[k][word[i][j][n] - 1] * Math.exp(mu[i][j][k]);
                        }
                        norm += gamma[i][j][n][k];
                    }
                    //ErrorHandler.catchNaNError(K, gamma[i][j][n], "Gamma");

                    for (int k = 0; k < K; k++) {
                        gamma[i][j][n][k] /= norm;
                        sum_gamma[k] += gamma[i][j][n][k];
                        //maintain mean_nkw mean_nkd
                        mean_nkw[k][word[i][j][n] - 1] += gamma[i][j][n][k] - prev_gamma[k];
                        mean_nkd[k] += gamma[i][j][n][k] - prev_gamma[k];
                    }
                }

                //update mu using Newton's method
                for (int k = 0; k < K; k++) {
                    double sum_mu = 0;
                    for (int d = 0; d < doc_word_slice[i][0]; d++)
                        if (d != j)
                            sum_mu += mu[i][d][k];
                    double Const = (Alpha - 1) / (double) doc_word_slice[i][0] + sum_gamma[k]
                            + sum_mu / ((double) doc_word_slice[i][0] * Sigma);

                    mu[i][j][k] = NewtonsMethodforMu(i, j, k, 1, 8, Const);

                    //Detect -Infinity Error
                    if (mu[i][j][k] == Double.NEGATIVE_INFINITY) {
                        System.out.format("zeta[i][j]:%.10f doc_word_slice:%d sum_gamma:%.10f sigma:%.10f", zeta[i][j]
                                , doc_word_slice[i][j + 1]
                                , sum_gamma[k], sigma[i][j][k]);
                        System.out.println();
                        System.exit(0);
                    }
                }
                //ErrorHandler.catchNaNError(K, mu[i][j], "Mu");

                //update sigma using Newton's method
                for (int k = 0; k < K; k++) {
                    sigma[i][j][k] = NewtonsMethodforSigma(i, j, k, 0.00001, 5);
                }
                //ErrorHandler.catchNaNError(K, sigma[i][j], "Sigma");

                //update zeta
                double temp_zeta = 0;
                for (int k = 0; k < K; k++)
                    temp_zeta += Math.exp(mu[i][j][k] + sigma[i][j][k] / 2.0);
                zeta[i][j] = temp_zeta;
                //ErrorHandler.catchNaNError(doc_word_slice[i][0], zeta[i], "Zeta");
            }
        }

        System.out.println("Iterating... No: " + iter_No);

        //store phi
        if(flag_Phi)
            storePhi();
        if (iter_No == 500) {
            //store thetap and phi when iterate 500 times
            storePhi();
            storeThetap_normed();
            return true;
        }
        return false;
    }

    private void storePhi() throws Exception {
        PrintWriter writerphi = new PrintWriter(outputDir + "model_iter_" + iter_No + ".phi", "UTF-8");
        for(int k = 0; k < K; k ++) {
            for (int v = 0; v < V; v++)
                writerphi.print(Double.toString((Beta + mean_nkw[k][v]) / (V * Beta + mean_nkd[k])) + " ");
            writerphi.println();
        }
        writerphi.close();
    }

    private void storeThetap_normed() throws Exception {
        PrintWriter writer = new PrintWriter(outputDir + "model_iter_" + iter_No + ".thetap", "UTF-8");
            for(int s = 0; s < S; s ++)
                for(int d = 0; d < doc_word_slice[s][0]; d ++) {
                    double sum = 0;
                    for (int k = 0; k < K; k++)
                        sum += Math.exp(mu[s][d][k]);
                    for (int k = 0; k < K; k++)
                        writer.print(Double.toString(Math.exp(mu[s][d][k]) / sum) + " ");
                    writer.println();
                }
        writer.close();
    }

    private double NewtonsMethodforSigma(int s, int d, int k, double x0, int step) {
        double xp = x0, xa;
        double[] state = new double[step];
        double Const = (1.0 - (double) doc_word_slice[s][0]) / (2.0 * (double) doc_word_slice[s][0] * Sigma);
        for(int i = 0; i < step; i ++) {
            state[i] = xp;
            xa = xp - (Math.log(zeta[s][d]) - Math.log(4 * doc_word_slice[s][d + 1])
                    + Math.log(2.0 * Const + 1.0 / xp) - xp / 2.0 - mu[s][d][k])
                / (- 1.0 / (2 * Const * xp * xp + xp) - 0.5);

            //Detect Error
            if(xa < 0 || xa != xa) {
                System.out.println("Error occur when updating sigma!..");
                System.out.format("xa:%f xp:%f zeta[s][d]:%f mu[s][d][k]:%f Const:%f doc_word_slice[s][d + 1]:%d \n", xa, xp
                        , zeta[s][d], mu[s][d][k], Const, doc_word_slice[s][d + 1]);
                for(int j = 0; j <= i; j ++)
                    System.out.format("%.10f ", state[j]);
                System.exit(0);
            }
            xp = xa;
        }
        return xp;
    }

    private double NewtonsMethodforMu(int s, int d, int k, double x0, int step, double Const) {
        //fix the crash(two cases) when exp{..} is too large or log{negative}
        boolean method_flag = true;
        if(Const + (1.0 - (double) doc_word_slice[s][0]) / (double) doc_word_slice[s][0] * x0 < 0) {
            method_flag = false;
        }
        double xp = x0, xa;
        double[] state = new double[step];
        for(int i = 0; i < step; i ++) {
            state[i] = xp;
            //divided into two cases to improve robustness
            if(method_flag) {
                xa = xp - (Math.log(Const + (1.0 - (double) doc_word_slice[s][0]) / (double) doc_word_slice[s][0] * xp)
                        + Math.log(zeta[s][d]) - Math.log(doc_word_slice[s][d + 1]) - xp - sigma[s][d][k] / 2.0)
                        / ((1.0 - (double) doc_word_slice[s][0]) / ((double) doc_word_slice[s][0] * Const
                        + (1.0 - (double) doc_word_slice[s][0]) * xp) - 1.0);
            }
            else {
                double exp_temp = (double) doc_word_slice[s][d + 1] / zeta[s][d] * Math.exp(xp + sigma[s][d][k] / 2.0);
                xa = xp - (Const + (1.0 - (double) doc_word_slice[s][0]) / (double) doc_word_slice[s][0] * xp
                                 - exp_temp)
                        / ((1.0 - (double) doc_word_slice[s][0]) / (double) doc_word_slice[s][0]
                                - exp_temp);
            }
            //Detect Error
            if(xa != xa) {
                System.out.println("Error occur when updating mu!..");
                System.out.format("xa:%f xp:%f Const:%f sigma[s][d][k]:%.20f zeta[s][d]:%.20f doc_word_slice[s][0]:%d doc_word_slice[s][d + 1]:%d \n"
                        , xa, xp, Const, sigma[s][d][k], zeta[s][d], doc_word_slice[s][0], doc_word_slice[s][d + 1]);
                for(int j = 0; j <= i; j ++)
                    System.out.format("%.10f ", state[j]);
                System.exit(0);
            }
            xp = xa;
        }
        return xp;
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

    public Map<String, Integer> getWordMap() {
        return wordMap;
    }
}
