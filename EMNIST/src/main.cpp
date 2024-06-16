#include "Storage.h"
#include "PCA.h"
#include "SVM.h"

int main()
{   
    std::string outputCsvFile = "/home/fhtw_user/msvr/EMNIST/dataset/elbow.csv";
    int optimalComponents = 1000;
    Storage storage;

    PCA pca;

    // pca.calculateExplainedVariance(storage.trainData.origin.features, 784, outputCsvFile);

    pca.fit(storage.trainData.origin.features, optimalComponents);
    pca.transform(storage.trainData.origin.features, storage.trainData.transformed.features,"TrainData");
    pca.transform(storage.testData.origin.features, storage.testData.transformed.features,"TestData");

    storage.convertData();
    storage.splitValidation();

    SVM svm;
    std::cout << "Created SVM" << std::endl;

    svm.optimizeParameters(storage.trainData.transformed.trainSubset.features, storage.trainData.transformed.trainSubset.targets, 
                           storage.trainData.transformed.validateSubset.features, storage.trainData.transformed.validateSubset.targets);                         
    std::cout << "Optimized SVM" << std::endl;

    svm.train(storage.trainData.transformed.features, storage.trainData.targets);
    std::cout << "trained SVM" << std::endl;

    float accuracy = svm.evaluate(storage.testData.transformed.features, storage.testData.targets);
    // Output the result
    std::cout << "Iteration 1: ACC " << accuracy * 100 << "%" << std::endl;

    return 0;
};