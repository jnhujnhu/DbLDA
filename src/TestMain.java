import Core.DbLDA;
import Core.Perplexity;

/**
 * Created by Kevin on 25/04/2017.
 */
public class TestMain {

    public static void main(String[] args) {
        try{
            DbLDA dbLDA = new DbLDA(1.01, 0.01, 0.01,50, 1, "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/");
            dbLDA.initWordMap("/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/wordmap.mp");
//            dbLDA.initParametersForVU("/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/trainset.dat", 28, 0.5);
//            while(!dbLDA.iterateVariationalUpdate());

            for(int i = 1; i < 500; i ++) {
                Perplexity perplexity = new Perplexity(50, dbLDA.getWordMap()
                        , "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/model.thetap"
                        , "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/Phis2/model_iter_" + Integer.toString(i) + ".phi"
                        , "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/trainset.dat");
                perplexity.importParameters();
                System.out.format("%d : %.15f \n", i, perplexity.evaluatePerplexity(0.5));
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

