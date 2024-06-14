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
    void optimizeParameters(const cv::Mat& trainData, const cv::Mat& trainLabels);

private:
    cv::Ptr<cv::ml::SVM> svm;
};