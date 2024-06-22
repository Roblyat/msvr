#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>
#include <random>
#include <opencv2/ml.hpp>
#include "string.h"
#include <numeric>

#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>
#include <random>

class Storage
{
private:
    // Loads the data from a CSV file
    void loadData();
    
    // Extracts specific rows for letters
    void extractRowsForLetters();
    
    // Prints the first and last few rows of the targets matrix
    void printRows(cv::Mat targets) const;
    
    // Prints the min and max values of a matrix
    void printMinMax(const cv::Mat& mat) const;
    
    // Shuffles the data
    void shuffleData();
    
    // Splits the data into training and test sets
    void splitData();
    
    // Standardizes the data
    void standardizeData();

    // Matrix to store feature data
    cv::Mat features;
    
    // Matrix to store target data
    cv::Mat targets;

    // Struct to store training data
    struct TrainData
    {
        // Targets for training data
        cv::Mat targets;

        struct origin
        {
            // Original features for training data
            cv::Mat features;
        } origin;

        struct transformed
        {
            // Transformed features for training data
            cv::Mat features;

            struct trainSubset
            {
                // Features for training subset
                cv::Mat features;
                
                // Targets for training subset
                cv::Mat targets;
            } trainSubset;

            struct validateSubset
            {
                // Features for validation subset
                cv::Mat features;
                
                // Targets for validation subset
                cv::Mat targets;
            } validateSubset;

        } transformed;
    };

    // Struct to store test data
    struct TestData
    {
        // Targets for test data
        cv::Mat targets;
        
        struct origin
        {
            // Original features for test data
            cv::Mat features;
        } origin;

        struct transformed
        {
            // Transformed features for test data
            cv::Mat features;
        } transformed;
    };

public:
    // Constructor: Calls methods to load, process, and split the data
    Storage();
    
    // Destructor
    ~Storage() = default;

    // Converts the data to appropriate types
    void convertData();
    
    // Splits the training data into training and validation sets
    void splitValidation();

    // Instance of TrainData to store training data
    TrainData trainData;
    
    // Instance of TestData to store test data
    TestData testData;
};

// Constructor: Calls methods to load, process, and split the data
Storage::Storage()
{
    loadData();
    extractRowsForLetters();
    shuffleData();
    splitData();
    standardizeData();
}

// Loads the data from a CSV file
void Storage::loadData()
{
    // Load the data from the CSV file
    // First column is the target as a float
    cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::loadFromCSV("/home/fhtw_user/msvr/EMNIST/dataset/emnist_letters_merged.csv", 0, 0, 1);
    features = tdata->getTrainSamples();  // Get design matrix (features)
    targets = tdata->getTrainResponses(); // Get target values

    std::cout << "###   INPUT   ###" << std::endl;
    printRows(targets); // Print the first and last few rows of the targets
}

// Prints the first and last few rows of the targets matrix
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

// Extracts rows for specific letters (target values 5 and 19)
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

    features = filteredFeatures; // Update features with filtered features
    targets = filteredTargets; // Update targets with filtered targets

    std::cout << "###   EXTRACTED  ###" << std::endl;
    std::cout << "Filtered features rows: " << features.rows << std::endl;
    std::cout << "Filtered targets rows: " << targets.rows << std::endl;

    printRows(targets); // Print the first and last few rows of the filtered targets
}

// Shuffles the data
void Storage::shuffleData()
{
    // Create a combined matrix of features and targets for shuffling
    cv::Mat combinedData(features.rows, features.cols + 1, CV_32F);
    features.copyTo(combinedData.colRange(1, combinedData.cols)); // Copy features to combined data
    targets.copyTo(combinedData.col(0)); // Copy targets to combined data

    std::cout << "###   COMBINED   ###" << std::endl;
    std::cout << "Combined data rows: " << combinedData.rows << std::endl;
    std::cout << "Combined data cols: " << combinedData.cols << std::endl;
    std::cout << "Label 0: " << combinedData.at<float>(0, 0) << std::endl;
    std::cout << "Label 799: " << combinedData.at<float>(799, 0) << std::endl;
    std::cout << "Label 800: " << combinedData.at<float>(800, 0) << std::endl;
    std::cout << "Label 1599: " << combinedData.at<float>(1599, 0) << std::endl;

    // Create a vector of row indices and shuffle them
    std::vector<int> indices(combinedData.rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Reorder the combined data according to the shuffled indices
    cv::Mat shuffledData(combinedData.rows, combinedData.cols, combinedData.type());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        combinedData.row(indices[i]).copyTo(shuffledData.row(i));
    }

    // Split the shuffled data back into features and targets
    shuffledData.colRange(1, shuffledData.cols).copyTo(features);
    shuffledData.col(0).copyTo(targets);
}

// Splits the data into training and test sets
void Storage::splitData()
{
    size_t trainSize = features.rows / 6; // 1/6 of the data for training

    // Split data into training and test sets
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
    printRows(trainData.targets); // Print the first and last few rows of the shuffled training targets
    std::cout << "###  shuffled test Data   ###: " << std::endl;
    printRows(testData.targets); // Print the first and last few rows of the shuffled test targets
}

// Prints the min and max values of a matrix
void Storage::printMinMax(const cv::Mat &mat) const
{
    double minVal, maxVal;
    cv::minMaxLoc(mat, &minVal, &maxVal);
    std::cout << "Min value: " << minVal << ", Max value: " << maxVal << std::endl;
}

// Standardizes the data
void Storage::standardizeData()
{
    std::cout << "###   BEFORE STANDARDIZATION   ###" << std::endl;
    printMinMax(features); // Print the min and max values of the features before standardization

    // Calculate mean and standard deviation for standardization
    cv::Mat mean, stddev;
    cv::meanStdDev(trainData.origin.features, mean, stddev);

    // Center and scale the features matrix
    cv::Mat features_centered_train = trainData.origin.features.clone();
    cv::Mat features_centered_test = testData.origin.features.clone();
    for (int i = 0; i < features.cols; ++i)
    {   
        features_centered_train.col(i) -= mean.at<double>(0, 0); // Center the training features
        features_centered_test.col(i) -= mean.at<double>(0, 0); // Center the test features
    }

    // Scale the features to avoid numerical issues
    for (int i = 0; i < features.cols; ++i)
    {
        double scale = stddev.at<double>(0, 0); // Scale factor
        if (scale < 1e-6)
        {
            scale = 1e-6;
        }
        features_centered_train.col(i) /= scale; // Scale the training features
        features_centered_test.col(i) /= scale; // Scale the test features
    }

    trainData.origin.features = features_centered_train; // Update training features with standardized data
    testData.origin.features = features_centered_test; // Update test features with standardized data

    // Recompute mean and stddev for standardized features
    cv::meanStdDev(trainData.origin.features, mean, stddev);
    std::cout << "###   STANDARDIZED   ###" << std::endl;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Stddev: " << stddev << std::endl;
    printMinMax(trainData.origin.features); // Print the min and max values of the standardized training features

    cv::meanStdDev(testData.origin.features, mean, stddev);
    std::cout << "###   STANDARDIZED   ###" << std::endl;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Stddev: " << stddev << std::endl;
    printMinMax(testData.origin.features); // Print the min and max values of the standardized test features
}

// Converts the data to appropriate types
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

// Splits the training data into training and validation sets
void Storage::splitValidation()
{
    // Calculate validation set size (20% of the data)
    int validationSize = trainData.transformed.features.rows / 5;
    
    // Split the transformed training data into training and validation subsets
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
    // Default constructor
    PCA() = default;

    // Default destructor
    ~PCA() = default;

    // Method to fit PCA on the given data with a specified number of components
    void fit(const cv::Mat& data, int numComponents);

    // Method to transform the given data using the fitted PCA model
    cv::Mat transform(const cv::Mat& data, cv::Mat &projectedData, const std::string& dataType);

    // Method to calculate explained variance for a specified maximum number of components
    // and save the results to a CSV file
    void calculateExplainedVariance(const cv::Mat &data, int maxComponents, const std::string &outputCsvFile);

private:
    // OpenCV PCA object to perform PCA operations
    cv::PCA pca;

    // Vector to store explained variances
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

/**
 * The calculateExplainedVariance method is used for creating an elbow plot.
 * An elbow plot is a graphical representation used to determine the optimal number of principal components
 * to retain in PCA. The x-axis represents the number of components, while the y-axis shows the cumulative
 * explained variance. The "elbow" point, where the explained variance starts to level off, indicates the
 * optimal number of components to use.
 */
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
    // Constructor: Initializes the SVM with default parameters
    SVM();
    
    // Default destructor
    ~SVM() = default;

    // Method to train the SVM on the given training data and labels
    void train(const cv::Mat& trainData, const cv::Mat& trainLabels);

    // Method to evaluate the SVM on the given test data and labels
    float evaluate(const cv::Mat& testData, const cv::Mat& testLabels);

    // Method to optimize SVM parameters (nu and gamma) using training and validation data
    void optimizeParameters(const cv::Mat& trainData, const cv::Mat& trainLabels, const cv::Mat& valData, const cv::Mat& valLabels);

private:
    // Pointer to the SVM object
    cv::Ptr<cv::ml::SVM> svm;
    
    // Pointer to the training data object
    cv::Ptr<cv::ml::TrainData> trainData;
    
    // Pointer to the test data object
    cv::Ptr<cv::ml::TrainData> testData;
    
    // Variables to store the best parameters and accuracy
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
    // Set the best parameters found during optimization
    svm->setNu(bestNu);
    svm->setGamma(bestGamma);

    // Create the training data
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
    // Define the range of nu and gamma values to search over
    std::vector<float> nuValues = {0.01, 0.1, 0.2, 0.3};
    std::vector<float> gammaValues = {0.0001, 0.001, 0.01, 0.1};

    // Iterate over a range of nu and gamma values to find the best combination
    for (double nu = 0.01; nu <= 0.5; nu += 0.01)
    {
        for (double gamma = 0.0001; gamma <= 0.001; gamma += 0.0001)
        {
            // Set the current nu and gamma values
            svm->setNu(nu);
            svm->setGamma(gamma);

            // Train the SVM with the current parameters
            train(trainData, trainLabels);

            // Evaluate the SVM on the validation data
            float accuracy = evaluate(valData, valLabels);
            std::cout << "Nu: " << nu << ", Gamma: " << gamma << ", Accuracy: " << accuracy * 100 << "%" << std::endl;

            // Update the best parameters if the current accuracy is better
            if (accuracy > bestAccuracy)
            {
                bestNu = nu;
                bestGamma = gamma;
                bestAccuracy = accuracy;
            }
        }
    }

    // Set the SVM to use the best parameters found during optimization
    svm->setNu(bestNu);
    svm->setGamma(bestGamma);
    std::cout << "Best Nu: " << bestNu << ", Best Gamma: " << bestGamma << ", Best Accuracy: " << bestAccuracy * 100 << "%" << std::endl;
}

int main()
{   
    // Path to the output CSV file for the elbow plot
    std::string outputCsvFile = "/home/fhtw_user/msvr/EMNIST/dataset/elbow.csv";
    
    // Number of optimal components for PCA
    int optimalComponents = 1000;
    
    // Create an instance of the Storage class to handle data loading and processing
    Storage storage;

    // Create an instance of the PCA class to perform PCA operations
    PCA pca;

    // Uncomment the following line to calculate explained variance for the elbow plot
    // pca.calculateExplainedVariance(storage.trainData.origin.features, 784, outputCsvFile);

    // Fit PCA on the training data with the specified number of components
    pca.fit(storage.trainData.origin.features, optimalComponents);
    
    // Transform the training and test data using the fitted PCA model
    pca.transform(storage.trainData.origin.features, storage.trainData.transformed.features, "TrainData");
    pca.transform(storage.testData.origin.features, storage.testData.transformed.features, "TestData");

    // Convert the data to the appropriate type
    storage.convertData();
    
    // Split the training data into training and validation sets
    storage.splitValidation();

    // Create an instance of the SVM class
    SVM svm;
    std::cout << "Created SVM" << std::endl;

    // Optimize SVM parameters using the training and validation data
    svm.optimizeParameters(storage.trainData.transformed.trainSubset.features, storage.trainData.transformed.trainSubset.targets, 
                           storage.trainData.transformed.validateSubset.features, storage.trainData.transformed.validateSubset.targets);                         
    std::cout << "Optimized SVM" << std::endl;

    // Train the SVM using the transformed training data and targets
    svm.train(storage.trainData.transformed.features, storage.trainData.targets);
    std::cout << "trained SVM" << std::endl;

    // Evaluate the SVM on the transformed test data
    float accuracy = svm.evaluate(storage.testData.transformed.features, storage.testData.targets);
    
    // Output the result
    std::cout << "Iteration 1: ACC " << accuracy * 100 << "%" << std::endl;

    return 0;
};

