#include "Storage.h"
#include "PCA.h"
#include "SVM.h"

int main()
{   
    std::string outputCsvFile = "/home/fhtw_user/msvr/EMNIST/dataset/elbow.csv";
    int optimalComponents = 50;
    Storage storage;

    storage.convertData();

    PCA pca;

    pca.calculateExplainedVariance(storage.trainData.origin.features, 50, outputCsvFile);

    pca.fit(storage.trainData.origin.features, optimalComponents);
    pca.transform(storage.trainData.origin.features, "TrainData");
    pca.transform(storage.testData.origin.features, "TestData");

    storage.convertData();

    SVM svm;
    std::cout << "Created SVM" << std::endl;
    svm.optimizeParameters(storage.trainData.transoformed.features, storage.trainData.targets);
    std::cout << "Optimized SVM" << std::endl;
    svm.train(storage.trainData.transoformed.features, storage.trainData.targets);
    std::cout << "trained SVM" << std::endl;
    float accuracy = svm.evaluate(storage.testData.transformed.features, storage.testData.targets);

    // Output the result
    std::cout << "Iteration 1: ACC " << accuracy * 100 << "%" << std::endl;

    return 0;
};