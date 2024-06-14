#include "Storage.h"
#include "PCA.h"
#include "SVM.h"

int main()
{
    Storage storage;

    PCA pca(50);
    pca.fit(storage.trainData.origin.features);
    pca.transform(storage.trainData.origin.features, "TrainData");
    pca.transform(storage.testData.origin.features, "TestData");

    SVM svm;

    svm.optimizeParameters(storage.trainData.transoformed.features, storage.trainData.targets);
    svm.train(storage.trainData.transoformed.features, storage.trainData.targets);
    float accuracy = svm.evaluate(storage.testData.transformed.features, storage.testData.targets);

    // Output the result
    std::cout << "Iteration 1: ACC " << accuracy * 100 << "%" << std::endl;


    return 0;
};