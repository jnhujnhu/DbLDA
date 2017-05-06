import Core.DbLDA;
import Core.Perplexity;

/**
 * Created by Kevin on 25/04/2017.
 */
public class TestMain {

    public static void main(String[] args) {
        try{
            //training
//            DbLDA dbLDA = new DbLDA(1.01, 0.01, 1,50, 1, "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/NewPhis/", "Test");
//            dbLDA.initWordMap("/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/wordmap.mp");
//            dbLDA.initParametersForVU("/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/reuters.dat", 1, 1);
//            //dbLDA.givenPhi("/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/NewPhis/model_iter_500.phi");
//            while(!dbLDA.iterateVariationalUpdate());

            //generate test theta
//            DbLDA dbLDA = new DbLDA(1.01, 0.01, 1,50, 1, "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/", "Test");
//            dbLDA.initWordMap("/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/wordmap.mp");
//            dbLDA.initParametersForVU("/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/trainset.dat", 28, 0.5);
//            dbLDA.givenPhi("/Users/Kevin/Desktop/Laboratory/1Problem/TestCorpus/NewPhis/model_iter_500.phi");
//            while(!dbLDA.iterateVariationalUpdate());

            //perplexity
            DbLDA dbLDA = new DbLDA(1.01, 0.01, 1,50, 1, "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/", "Test");
            dbLDA.initWordMap("/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/wordmap.mp");
            //for(int i = 1; i < 500; i ++) {
                Perplexity perplexity = new Perplexity(50, dbLDA.getWordMap()
                        , "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/model_iter_Test.thetap"
                        , "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/NewPhis/model_iter_500.phi"
                        , "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/trainset.dat"
                        , 0.5);
                perplexity.importParameters();
                System.out.format("%d : %.15f \n", 500, perplexity.evaluatePerplexity());
            //}

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

