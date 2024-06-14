#include "PCA.h"

PCA::PCA(int numComponents)
{
    this->numComponents = numComponents;
}

void PCA::fit(const cv::Mat& data)
{   
    // Ensure the data type is CV_32F
    cv::Mat floatData;
    data.convertTo(floatData, CV_32F);
    // Fit PCA on the training data
    pca = cv::PCA(floatData, cv::Mat(), cv::PCA::DATA_AS_ROW, numComponents);
    std::cout << "PCA fitted with " << numComponents << " components." << std::endl;
}

cv::Mat PCA::transform(const cv::Mat& data, const std::string& dataType)
{
    // Ensure the data type is CV_32F
    cv::Mat floatData;
    data.convertTo(floatData, CV_32F);

    // Transform the data using the fitted PCA model
    cv::Mat projectedData;
    pca.project(floatData, projectedData);
    std::cout << "TRANSFORMED" << dataType << std::endl;
    std::cout << "Transformed training data size: " << projectedData.size() << std::endl;
    return projectedData;
}
