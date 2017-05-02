package Core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Map;
import java.util.Scanner;

/**
 * Created by Kevin on 02/05/2017.
 */
public class Perplexity {
    private String thetaPath, phiPath, trainsetPath;

    private double[][] theta, phi;
    private int[][] word;
    private int[] doc_length;
    private int total_word;
    private int D, V, mD, K;

    private Map<String, Integer> wordmap;

    public Perplexity(int K, Map<String, Integer> wordMap, String thetaPath, String phiPath, String trainsetPath) {
        this.thetaPath = thetaPath;
        this.phiPath = phiPath;
        this.trainsetPath = trainsetPath;
        this.wordmap = wordMap;
        this.K = K;
        D = 0;
        V = wordMap.size();
        total_word = 0;
    }

    public void setPhiPath(String phiPath) {
        this.phiPath = phiPath;
    }

    private void readFromFile(double[][] para, String path) throws Exception {
        BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(new File(path))));
        Scanner scanner = new Scanner(input);
        int _no = 0;
        while(scanner.hasNextLine()) {
            String[] _temp = scanner.nextLine().split(" ");
            for(int i = 0; i < _temp.length; i ++) {
                para[_no][i] = Double.parseDouble(_temp[i]);
            }
            _no ++;
        }
        scanner.close();
    }

    public void importParameters() throws Exception {
        BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(new File(trainsetPath))));
        Scanner scanner = new Scanner(input);
        String[] para_temp = scanner.nextLine().split(" ");
        D = Integer.parseInt(para_temp[1]);
        mD = Integer.parseInt(para_temp[3]);

        word = new int[D][mD];
        theta = new double[D][K];
        phi = new double[K][V];
        doc_length = new int[D];

        readFromFile(theta, thetaPath);
        readFromFile(phi, phiPath);

        int doc_no = 0;
        while(scanner.hasNextLine()) {
            String[] word_temp = scanner.nextLine().split(" ");
            doc_length[doc_no] = word_temp.length - 1;
            total_word += doc_length[doc_no];
            for(int j = 1; j < word_temp.length; j ++)
                word[doc_no][j - 1] = wordmap.get(word_temp[j]);
            doc_no ++;
        }

        scanner.close();
    }

    public double evaluatePerplexity(double doc_percent) {
        double log_sum = 0;
        for(int i = 0; i < D; i ++) {
            for(int j = (int) Math.floor(doc_length[i] * doc_percent); j < doc_length[i]; j ++) {
                double temp_ = 0;
                for(int k = 0; k < K; k ++) {
                    temp_ += theta[i][k] * phi[k][word[i][j] - 1];
                }
                log_sum += Math.log(temp_);
            }
        }
        return Math.exp(- log_sum / (double) total_word);
    }
}
