package weka.app;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.NewTree;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class TestingApp {

    public static void main(String[] args) throws Exception {
        String rootPath="C:\\Users\\Przemek\\Desktop\\male\\baza_01_train.arff";
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(rootPath));
        Instances originalTrain = loader.getDataSet();
        originalTrain.setClassIndex(originalTrain.numAttributes() - 1);
        NewTree newTree = new NewTree();

        newTree.setUniquePairs(true);
        //newTree.setUniquePairs(true);
        newTree.setKTopScoringPairs(2);
        newTree.setMaxDepth(100);


        newTree.buildClassifier(originalTrain);
        System.out.println(newTree.toString());

        loader.setFile(new File("C:\\Users\\Przemek\\Desktop\\male\\baza_01_test.arff"));
        Instances testData = loader.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1);

        Evaluation ev = new Evaluation(originalTrain);
        Random rand = new Random(1);
        int folds = 10;
        ev.crossValidateModel(newTree, testData, folds, rand);
        System.out.println(ev.toSummaryString());
    }
}
