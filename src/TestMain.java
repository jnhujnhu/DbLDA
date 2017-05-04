import Core.DbLDA;
import Core.Perplexity;

/**
 * Created by Kevin on 25/04/2017.
 */
public class TestMain {

    public static void main(String[] args) {
        try{
            DbLDA dbLDA = new DbLDA(1.01, 0.01, 0.01,50, 1, "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/Testonly/");
            dbLDA.initWordMap("/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/wordmap.mp");
            dbLDA.initParametersForVU("/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/reuters.dat", 1, 1);
//            dbLDA.givenPhi("/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/Phis2/model_iter_500.phi");
            while(!dbLDA.iterateVariationalUpdate());

            //for(int i = 1; i < 500; i ++) {
//                Perplexity perplexity = new Perplexity(50, dbLDA.getWordMap()
//                        , "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/Testonly/model_iter_14.thetap"
//                        , "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/Testonly/model_iter_14.phi"
//                        , "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/reuters.dat"
//                        , 0.5);
//                perplexity.importParameters();
//                System.out.format("%d : %.15f \n", 1, perplexity.evaluatePerplexity());
            //}

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

