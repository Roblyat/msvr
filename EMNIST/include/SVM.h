#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

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