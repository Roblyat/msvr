#include "SVM.h"

SVM::SVM()
{
    // Initialize the SVM with default parameters
    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::NU_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
}

void SVM::train(const cv::Mat& features, const cv::Mat& lables) //######## HIER TRAIN SEBLER GRID SEARCH MACHEN COCO CPP
{
    // Ensure the data type is CV_32F
    // trainData.convertTo(trainData, CV_32F);
    // trainLabels.convertTo(trainLabels, CV_32S);
    svm->setNu(0.01);
    svm->setGamma(0.0001);
    this->trainData = cv::ml::TrainData::create(features, cv::ml::ROW_SAMPLE, lables);
    // Train the SVM
    svm->train(trainData, cv::ml::ROW_SAMPLE);
}

float SVM::evaluate(const cv::Mat& testData, const cv::Mat& testLabels)
{
    // Ensure the data type is CV_32F
    // testData.convertTo(testData, CV_32F);
    // testLabels.convertTo(testLabels, CV_32S);

    // Evaluate the SVM on the test data
    cv::Mat predictions;
    svm->predict(testData, predictions);

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < testLabels.rows; ++i)
    {
        if (predictions.at<float>(i, 0) == testLabels.at<float>(i, 0))
        {
            correct++;
        }
    }
    return static_cast<float>(correct) / testLabels.rows;
}

void SVM::optimizeParameters(const cv::Mat& trainData, const cv::Mat& trainLabels)
{
    // Ensure the data type is CV_32F
    // trainData.convertTo(trainData, CV_32F);
    // trainLabels.convertTo(trainLabels, CV_32S);

    // Optimize parameters using grid search and cross-validation
    cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, trainLabels);
    svm->trainAuto(tdata);
}
