#include "SVM.h"

SVM::SVM()
{
    // Initialize the SVM with default parameters
    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
}

void SVM::train(const cv::Mat& trainData, const cv::Mat& trainLabels)
{
    // Ensure the data type is CV_32F
    cv::Mat floatTrainData, floatTrainLabels;
    trainData.convertTo(floatTrainData, CV_32F);
    trainLabels.convertTo(floatTrainLabels, CV_32F);

    // Train the SVM
    svm->train(floatTrainData, cv::ml::ROW_SAMPLE, floatTrainLabels);
}

float SVM::evaluate(const cv::Mat& testData, const cv::Mat& testLabels)
{
    // Ensure the data type is CV_32F
    cv::Mat floatTestData, floatTestLabels;
    testData.convertTo(floatTestData, CV_32F);
    testLabels.convertTo(floatTestLabels, CV_32F);

    // Evaluate the SVM on the test data
    cv::Mat predictions;
    svm->predict(floatTestData, predictions);

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < floatTestLabels.rows; ++i)
    {
        if (predictions.at<float>(i, 0) == floatTestLabels.at<float>(i, 0))
        {
            correct++;
        }
    }
    return static_cast<float>(correct) / floatTestLabels.rows;
}

void SVM::optimizeParameters(const cv::Mat& trainData, const cv::Mat& trainLabels)
{
    // Ensure the data type is CV_32F
    cv::Mat floatTrainData, floatTrainLabels;
    trainData.convertTo(floatTrainData, CV_32F);
    trainLabels.convertTo(floatTrainLabels, CV_32F);

    // Optimize parameters using grid search and cross-validation
    cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(floatTrainData, cv::ml::ROW_SAMPLE, floatTrainLabels);
    svm->trainAuto(tdata);
}
