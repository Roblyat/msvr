#pragma once
#include "string.h"
#include <opencv2/opencv.hpp>
#include <iostream>

class PCA
{
public:
    PCA(int numComponents);
    ~PCA() = default;

    void fit(const cv::Mat& data);
    cv::Mat transform(const cv::Mat& data, const std::string& dataType);

private:
    cv::PCA pca;
    int numComponents;
};