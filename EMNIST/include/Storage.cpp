// Storage.cpp
#include "Storage.h"

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
        trainData.features.push_back(features.row(i));
        trainData.targets.push_back(targets.row(i));
    }

    for (size_t i = trainSize; i < static_cast<size_t>(features.rows); ++i)
    {
        testData.features.push_back(features.row(i));
        testData.targets.push_back(targets.row(i));
    }
    std::cout << "###   SHUFFLED AND SPLIT   ###" << std::endl;
    std::cout << "Train data features rows: " << trainData.features.rows << std::endl;
    std::cout << "Train data targets rows: " << trainData.targets.rows << std::endl;
    std::cout << "Test data features rows: " << testData.features.rows << std::endl;
    std::cout << "Test data targets rows:  " << testData.targets.rows << std::endl;

    std::cout << "###   SHUFFLE TEST  ###" << std::endl;
    std::cout << "###   shuffled train Data   ###: " << std::endl;
    printRows(trainData.targets);
    std::cout << "###  shuffled test Data   ###: " << std::endl;
    printRows(testData.targets);
}

void Storage::standardizeData()
{
    // Standardize the features matrix
    cv::Mat mean, stddev;
    cv::meanStdDev(trainData.features, mean, stddev);

    // Initialisiere X_standardized mit den gleichen Dimensionen und Typ wie features
    cv::Mat features_centered = trainData.features.clone();
    for (int i = 0; i < trainData.features.cols; ++i) {
        features_centered.col(i) -= mean.at<double>(i);
    }

    // Numerische Probleme vermeiden und standardisieren
    for (int i = 0; i < trainData.features.cols; ++i) {
        double scale = stddev.at<double>(i);
        if (scale < 1e-6) {
            scale = 1e-6;
        }
        features_centered.col(i) /= scale;
    }

    trainData.features = features_centered;

    cv::meanStdDev(trainData.features, mean, stddev);
    std::cout << "###   STANDARDIZED   ###" << std::endl;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Stddev: " << stddev << std::endl;
}