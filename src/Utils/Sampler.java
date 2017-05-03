package Utils;

import org.apache.commons.math3.distribution.GammaDistribution;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Created by Kevin on 03/05/2017.
 */
public class Sampler {

    public static double[] getGaussianSample(int K, double[] mean, double[] Deviation) {
        Random r = new Random();
        double[] sample = new double[K];
        for(int k = 0; k < K; k ++) {
            sample[k] = r.nextGaussian() * Math.sqrt(Deviation[k]) + mean[k];
        }
        return sample;
    }

    public static double[] getDirichletSample(int K, double Alpha) {
        double[] sample = new double[K];
        double norm = 0;
        for(int k = 0; k < K; k ++) {
            sample[k] = new GammaDistribution(Alpha, 1).sample();
            norm += sample[k];
        }
        for(int k = 0; k < K; k ++)
            sample[k] /= norm;
        return sample;
    }

    public static int getMultinomialSample(double[] parameters) {
        Random r = new Random();
        Arrays.sort(parameters);
        double sample = r.nextDouble();
        double temp = 0;
        for(int i = 0; i < parameters.length; i ++) {
            temp += parameters[i];
            if(sample <= temp)
                return i;
        }
        return -1;
    }
}
