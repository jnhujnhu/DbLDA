import Core.DbLDA;
import Core.Perplexity;

/**
 * Created by Kevin on 25/04/2017.
 */
public class TestMain {

    public static void main(String[] args) {
        try{
            DbLDA dbLDA = new DbLDA(1, 0.01, 0,50, 1, "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/Phis2/");
            dbLDA.initWordMap("/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/wordmap.mp");
//            dbLDA.initParametersForVU("/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/reuters.dat", 1, 1);
//            while(!dbLDA.iterateVariationalUpdate());

            Perplexity perplexity = new Perplexity(50, dbLDA.getWordMap()
                    , "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/train_parameter/model.thetap"
                    , "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/Phis/model_iter_500.phi"
                    , "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/trainset.dat");
            perplexity.importParameters();
            System.out.println(perplexity.evaluatePerplexity(0.5));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

