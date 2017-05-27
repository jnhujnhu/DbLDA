import Core.DbLDA;
import Core.Perplexity;

import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.io.File;

/**
 * Created by Kevin on 25/04/2017.
 */
public class TestMain {

    static void writePerplexity(double perplexity, String Home_Dir) throws Exception{
        PrintWriter writer = new PrintWriter(new FileOutputStream(new File(Home_Dir + "DbLDA_Perplexity(2_30).txt"), true));
        writer.println(perplexity);
        writer.close();
    }

    public static void main(String[] args) {
        try {

            //corpus directory prefix
            String Home_Dir = "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/";

            //Training Phi(topic-word matrix)
            DbLDA dbLDA = new DbLDA(1.01, 0.01, 1, 50, 7, Home_Dir + "DbLDA_train/", true);
            //init word map
            dbLDA.initWordMap(Home_Dir + "wordmap.mp");
            //random initialize parameters(mu, sigma, zeta, gamma)
            dbLDA.initParametersForVU(Home_Dir + "reuters.dat", 1, 1);

            long startTime = System.currentTimeMillis();
            //iterate update
            while (!dbLDA.iterateVariationalUpdate()) {
                System.out.print(dbLDA.getCurrentIterateCount());
                System.out.print(" ");
                System.out.println(System.currentTimeMillis() - startTime);
            }

            //Perplexity Test over Test Set
            //generate test theta(topic proportion of documents in test set), same learning process
            DbLDA dbLDA_test = new DbLDA(1.01, 0.01, 1, 50, 7, Home_Dir + "DbLDA_test/", false);
            dbLDA_test.initWordMap(Home_Dir + "wordmap.mp");
            //using first half of documents in test set to generate theta
            dbLDA_test.initParametersForVU(Home_Dir + "reuters_testset.dat", 28, 0.5);
            //using the model(phi) learned from training set
            dbLDA_test.givenPhi(Home_Dir + "DbLDA_train/model_iter_500.phi");
            while (!dbLDA_test.iterateVariationalUpdate()) ;

            //evaluate perplexity using latter half of test set
            Perplexity perplexity = new Perplexity(50, dbLDA.getWordMap()
                    , Home_Dir + "DbLDA_test/model_iter_500.thetap"
                    , Home_Dir + "DbLDA_train/model_iter_500.phi"
                    , Home_Dir + "trainset.dat"
                    , 0.5);
            perplexity.importParameters();
            //output perplexity to file
            writePerplexity(perplexity.evaluatePerplexity(), Home_Dir);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

