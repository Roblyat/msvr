#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>
#include <random>
#include <opencv2/ml.hpp>
#include "string.h"
#include <numeric>

class Storage
{
private:
    void loadData();
    void extractRowsForLetters();
    void printRows(cv::Mat targets) const;
    void printMinMax(const cv::Mat& mat) const;
    void shuffleData();
    void splitData();
    void standardizeData();

    cv::Mat features;
    cv::Mat targets;

    struct TrainData
    {
        cv::Mat targets;

        struct origin
        {
            cv::Mat features;
        }origin;

        struct transformed
        {
            cv::Mat features;

            struct trainSubset
            {
                cv::Mat features;
                cv::Mat targets;
            }trainSubset;

            struct validateSubset
            {
                cv::Mat features;
                cv::Mat targets;
            }validateSubset;

        }transformed;
    };

    struct TestData
    {
        cv::Mat targets;
        
        struct origin
        {
            cv::Mat features;
        }origin;

        struct transformed
        {
            cv::Mat features;
        }transformed;
    };

public:
    Storage();
    ~Storage() = default;

    void convertData();
    void splitValidation();

    TrainData trainData;
    TestData testData;
};



Storage::Storage()
{
    loadData();
    extractRowsForLetters();
    shuffleData();
    splitData();
    standardizeData();
}

void Storage::loadData()
{
    // Load the data
    // First col is the target as a float
    cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::loadFromCSV("/home/fhtw_user/msvr/EMNIST/dataset/emnist_letters_merged.csv", 0, 0, 1);
    features = tdata->getTrainSamples();  // Get design matrix
    targets = tdata->getTrainResponses(); // Get target values

    std::cout << "###   INPUT   ###" << std::endl;
    printRows(targets);
}

void Storage::printRows(cv::Mat targets) const
{
    std::cout << "First five rows of targets:" << std::endl;
    for (int i = 0; i < 5; ++i)
    {
        std::cout << targets.at<float>(i, 0) << std::endl;
    }

    std::cout << "Last five rows of targets:" << std::endl;
    for (int i = targets.rows - 5; i < targets.rows; ++i)
    {
        std::cout << targets.at<float>(i, 0) << std::endl;
    }
}

void Storage::extractRowsForLetters()
{
    cv::Mat filteredFeatures, filteredTargets;
    for (int i = 0; i < targets.rows; ++i)
    {
        float target = targets.at<float>(i, 0);
        if (static_cast<int>(target) == 5 || static_cast<int>(target) == 19)
        {
            filteredFeatures.push_back(features.row(i));
            filteredTargets.push_back(targets.row(i));
        }
    }

    features = filteredFeatures;
    targets = filteredTargets;

    std::cout << "###   EXTRACTED  ###" << std::endl;
    std::cout << "Filtered features rows: " << features.rows << std::endl;
    std::cout << "Filtered targets rows: " << targets.rows << std::endl;

    printRows(targets);
}

void Storage::shuffleData()
{
    // Create a combined matrix of features and targets for shuffling
    cv::Mat combinedData(features.rows, features.cols + 1, CV_32F);
    features.copyTo(combinedData.colRange(1, combinedData.cols));
    targets.copyTo(combinedData.col(0));

    std::cout << "###   COMBINED   ###" << std::endl;
    std::cout << "Combined data rows: " << combinedData.rows << std::endl;
    std::cout << "Combined data cols: " << combinedData.cols << std::endl;
    std::cout << "Lable 0: " << combinedData.at<float>(0, 0) << std::endl;
    std::cout << "Lable 799: " << combinedData.at<float>(799, 0) << std::endl;
    std::cout << "Lable 800: " << combinedData.at<float>(800, 0) << std::endl;
    std::cout << "Lable 1599: " << combinedData.at<float>(1599, 0) << std::endl;

    // Shuffle the combined data
    // Create a vector of row indices
    std::vector<int> indices(combinedData.rows);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle the indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Create a new matrix to store the shuffled data
    cv::Mat shuffledData(combinedData.rows, combinedData.cols, combinedData.type());

    // Reorder the combined data according to the shuffled indices
    for (size_t i = 0; i < indices.size(); ++i)
    {
        combinedData.row(indices[i]).copyTo(shuffledData.row(i));
    }

    // Split the shuffled data back into features and targets
    shuffledData.colRange(1, shuffledData.cols).copyTo(features);
    shuffledData.col(0).copyTo(targets);
}

void Storage::splitData()
{
    size_t trainSize = features.rows / 6; // 1/6 of the data for training

    // Ensure that indices are shuffled and split according to the calculated sizes
    for (size_t i = 0; i < trainSize; ++i)
    {
        trainData.origin.features.push_back(features.row(i));
        trainData.targets.push_back(targets.row(i));
    }

    for (size_t i = trainSize; i < static_cast<size_t>(features.rows); ++i)
    {
        testData.origin.features.push_back(features.row(i));
        testData.targets.push_back(targets.row(i));
    }
    std::cout << "###   SHUFFLED AND SPLIT   ###" << std::endl;
    std::cout << "Train data features rows: " << trainData.origin.features.rows << std::endl;
    std::cout << "Train data targets rows: " << trainData.targets.rows << std::endl;
    std::cout << "Test data features rows: " << testData.origin.features.rows << std::endl;
    std::cout << "Test data targets rows:  " << testData.targets.rows << std::endl;

    std::cout << "###   SHUFFLE TEST  ###" << std::endl;
    std::cout << "###   shuffled train Data   ###: " << std::endl;
    printRows(trainData.targets);
    std::cout << "###  shuffled test Data   ###: " << std::endl;
    printRows(testData.targets);
}

void Storage::printMinMax(const cv::Mat &mat) const
{
    double minVal, maxVal;
    cv::minMaxLoc(mat, &minVal, &maxVal);
    std::cout << "Min value: " << minVal << ", Max value: " << maxVal << std::endl;
}

// Add this function call in standardizeData
void Storage::standardizeData()
{
    std::cout << "###   BEFORE STANDARDIZATION   ###" << std::endl;
    printMinMax(features);

    // Standardize the features matrix
    cv::Mat mean, stddev;
    cv::meanStdDev(trainData.origin.features, mean, stddev);

    // Initialize features_centered with the same dimensions and type as features
    cv::Mat features_centered_train = trainData.origin.features.clone();
    cv::Mat features_centered_test = testData.origin.features.clone();
    for (int i = 0; i < features.cols; ++i)
    {   
        features_centered_train.col(i) -= mean.at<double>(0, 0); // Use proper indexing
        features_centered_test.col(i) -= mean.at<double>(0, 0);
    }

    // Avoid numerical issues and standardize
    for (int i = 0; i < features.cols; ++i)
    {
        double scale = stddev.at<double>(0, 0); // Use proper indexing
        if (scale < 1e-6)
        {
            scale = 1e-6;
        }
        features_centered_train.col(i) /= scale;
        features_centered_test.col(i) /= scale;
    }

    trainData.origin.features = features_centered_train;
    testData.origin.features = features_centered_test;

    // Recompute mean and stddev for standardized features
    cv::meanStdDev(trainData.origin.features, mean, stddev);
    std::cout << "###   STANDARDIZED   ###" << std::endl;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Stddev: " << stddev << std::endl;
    printMinMax(trainData.origin.features);

    cv::meanStdDev(testData.origin.features, mean, stddev);
    std::cout << "###   STANDARDIZED   ###" << std::endl;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Stddev: " << stddev << std::endl;
    printMinMax(testData.origin.features);
}

void Storage::convertData()
{
    // Convert the data to the appropriate type
    trainData.origin.features.convertTo(trainData.origin.features, CV_32F);
    trainData.transformed.features.convertTo(trainData.transformed.features, CV_32F);
    trainData.targets.convertTo(trainData.targets, CV_32S);
    testData.origin.features.convertTo(testData.origin.features, CV_32F);
    testData.transformed.features.convertTo(testData.transformed.features, CV_32F);
    testData.targets.convertTo(testData.targets, CV_32S);

    std::cout << "###   CONVERTED   ###" << std::endl;
    std::cout << "Train data features type: " << trainData.origin.features.type() << std::endl;
    std::cout << "Train data targets type: " << trainData.targets.type() << std::endl;
    std::cout << "Test data features type: " << testData.origin.features.type() << std::endl;
    std::cout << "Test data targets type: " << testData.targets.type() << std::endl;
}

void Storage::splitValidation()
{
    // Split trainData into training and validation sets
    int validationSize = trainData.transformed.features.rows / 5; // 20% validation
    trainData.transformed.trainSubset.features = trainData.transformed.features.rowRange(0, trainData.transformed.features.rows - validationSize);
    trainData.transformed.trainSubset.targets = trainData.targets.rowRange(0, trainData.targets.rows - validationSize);
    trainData.transformed.validateSubset.features = trainData.transformed.features.rowRange(trainData.transformed.features.rows - validationSize, trainData.transformed.features.rows);
    trainData.transformed.validateSubset.targets = trainData.targets.rowRange(trainData.targets.rows - validationSize, trainData.targets.rows);

    std::cout << "###   SPLIT   ###" << std::endl;
    std::cout << "Train subset features rows: " << trainData.transformed.trainSubset.features.rows << std::endl;
    std::cout << "Train subset targets rows: " << trainData.transformed.trainSubset.targets.rows << std::endl;
    std::cout << "Validate subset features rows: " << trainData.transformed.validateSubset.features.rows << std::endl;
    std::cout << "Validate subset targets rows: " << trainData.transformed.validateSubset.targets.rows << std::endl;
    std::cout << "Train subset 5 first targets: " << trainData.transformed.trainSubset.targets.rowRange(0, 5) << std::endl;
    std::cout << "Validate subset 5 first targets: " << trainData.transformed.validateSubset.targets.rowRange(0, 5) << std::endl;   
}



class PCA
{
public:
    PCA() = default;
    ~PCA() = default;

    void fit(const cv::Mat& data, int numComponents);
    cv::Mat transform(const cv::Mat& data, cv::Mat &projectedData, const std::string& dataType);
    void calculateExplainedVariance(const cv::Mat &data, int maxComponents, const std::string &outputCsvFile);

private:
    cv::PCA pca;
    std::vector<double> explainedVariances;
};


void PCA::fit(const cv::Mat &data, int numComponents)
{
    // Ensure the data type is CV_32F
    data.convertTo(data, CV_32F);
    // Fit PCA on the training data
    pca = cv::PCA(data, cv::Mat(), cv::PCA::DATA_AS_ROW, numComponents);
    std::cout << "PCA fitted with " << numComponents << " components." << std::endl;
}

cv::Mat PCA::transform(const cv::Mat &data, cv::Mat &projectedData, const std::string &dataType)
{
    // Ensure the data type is CV_32F
    data.convertTo(data, CV_32F);

    // Transform the data using the fitted PCA model
    pca.project(data, projectedData);
    std::cout << "TRANSFORMED " << dataType << std::endl;
    std::cout << "Transformed " << dataType << " training data size: " << projectedData.size() << std::endl;

    return projectedData;
}

void PCA::calculateExplainedVariance(const cv::Mat &data, int maxComponents, const std::string &outputCsvFile)
{
    explainedVariances.clear(); // Clear previous variances

    // Calculate the total variance from all components
    double totalVariance = 0.0;
    cv::PCA pcaAll(data, cv::Mat(), cv::PCA::DATA_AS_ROW, data.cols);
    cv::Mat eigenvaluesAll = pcaAll.eigenvalues;
    for (int i = 0; i < eigenvaluesAll.rows; ++i)
    {
        totalVariance += eigenvaluesAll.at<float>(i, 0);
    }

    // Open the CSV file for writing
    std::ofstream csvFile(outputCsvFile);
    if (!csvFile.is_open())
    {
        throw std::runtime_error("Could not open file to save explained variances.");
    }

    // Write the header
    csvFile << "Component,Eigenvalue,ExplainedVariance,CumulativeVariance\n";

    for (int i = 1; i <= maxComponents; ++i)
    {
        std::cout << "Processing component " << i << " out of " << maxComponents << std::endl;
        cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, i);
        cv::Mat eigenvalues = pca.eigenvalues;

        double cumulativeVariance = 0.0;
        for (int j = 0; j < i; ++j)
        {
            double explainedVariance = eigenvalues.at<float>(j, 0) / totalVariance;
            cumulativeVariance += explainedVariance;
            csvFile << i << ","
                    << eigenvalues.at<float>(j, 0) << ","
                    << explainedVariance << ","
                    << cumulativeVariance << "\n";
        }
    }

    csvFile.close();
    std::cout << "###  Saved " << outputCsvFile << "   ###" << std::endl;
}


class SVM
{
public:
    SVM();
    ~SVM() = default;

    void train(const cv::Mat& trainData, const cv::Mat& trainLabels);
    float evaluate(const cv::Mat& testData, const cv::Mat& testLabels);
    void optimizeParameters(const cv::Mat& trainData, const cv::Mat& trainLabels, const cv::Mat& valData, const cv::Mat& valLabels);

private:
    cv::Ptr<cv::ml::SVM> svm;
    cv::Ptr<cv::ml::TrainData> trainData;
    cv::Ptr<cv::ml::TrainData> testData;
    float bestNu = 0.001;
    float bestGamma = 0.0001;
    float bestAccuracy = 0.0;
};


SVM::SVM()
{
    // Initialize the SVM with default parameters
    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::NU_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
}

void SVM::train(const cv::Mat& features, const cv::Mat& lables) 
{
    svm->setNu(bestNu);
    svm->setGamma(bestGamma);
    this->trainData = cv::ml::TrainData::create(features, cv::ml::ROW_SAMPLE, lables);
    // Train the SVM
    svm->train(trainData, cv::ml::ROW_SAMPLE);
}

float SVM::evaluate(const cv::Mat& testData, const cv::Mat& testLabels)
{
    // Ensure the data type is CV_32F
    cv::Mat floatTestData;
    testData.convertTo(floatTestData, CV_32F);

    // Ensure the labels are in CV_32S for comparison
    cv::Mat intTestLabels;
    testLabels.convertTo(intTestLabels, CV_32S);

    // Evaluate the SVM on the test data
    cv::Mat predictions;
    svm->predict(floatTestData, predictions);

    // Ensure predictions are in CV_32S for comparison
    cv::Mat intPredictions;
    predictions.convertTo(intPredictions, CV_32S);

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < intTestLabels.rows; ++i)
    {
        int predictedLabel = intPredictions.at<int>(i, 0);
        int actualLabel = intTestLabels.at<int>(i, 0);

        if (predictedLabel == actualLabel)
        {
            correct++;
        }
    }
    return static_cast<float>(correct) / intTestLabels.rows;
}

void SVM::optimizeParameters(const cv::Mat& trainData, const cv::Mat& trainLabels, const cv::Mat& valData, const cv::Mat& valLabels)
{
    std::vector<float> nuValues = {0.01, 0.1, 0.2, 0.3};
    std::vector<float> gammaValues = {0.0001, 0.001, 0.01, 0.1};

    for (double nu = 0.01; nu <= 0.5; nu += 0.01)
    {
        for (double gamma = 0.0001; gamma <= 0.001; gamma += 0.0001)
        {
            svm->setNu(nu);
            svm->setGamma(gamma);

            train(trainData, trainLabels);

            float accuracy = evaluate(valData, valLabels);
            std::cout << "Nu: " << nu << ", Gamma: " << gamma << ", Accuracy: " << accuracy * 100 << "%" << std::endl;

            if (accuracy > bestAccuracy)
            {
                bestNu = nu;
                bestGamma = gamma;
                bestAccuracy = accuracy;
            }
        }
    }

    svm->setNu(bestNu);
    svm->setGamma(bestGamma);
    std::cout << "Best Nu: " << bestNu << ", Best Gamma: " << bestGamma << ", Best Accuracy: " << bestAccuracy * 100 << "%" << std::endl;
}




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