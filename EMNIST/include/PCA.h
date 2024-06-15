#pragma once
#include "string.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>

class PCA
{
public:
    PCA() = default;
    ~PCA() = default;

    void fit(const cv::Mat& data, int numComponents);
    cv::Mat transform(const cv::Mat& data, const std::string& dataType);
    void calculateExplainedVariance(const cv::Mat &data, int maxComponents, const std::string &outputCsvFile);

private:
    cv::PCA pca;
    std::vector<double> explainedVariances;
};