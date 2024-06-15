#include "PCA.h"

void PCA::fit(const cv::Mat& data, int numComponents)
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
    std::cout << "TRANSFORMED " << dataType << std::endl;
    std::cout << "Transformed " << dataType <<" training data size: " << projectedData.size() << std::endl;
    return projectedData;
}

void PCA::calculateExplainedVariance(const cv::Mat &data, int maxComponents, const std::string &outputCsvFile)
{
    for (int i = 1; i <= maxComponents; ++i) {
        fit(data, i);
        cv::Mat eigenvalues = pca.eigenvalues;// Assuming you add a method to get eigenvalues
        double totalVariance = cv::sum(eigenvalues)[0];
        double explainedVariance = cv::sum(eigenvalues.rowRange(0, i))[0] / totalVariance;
        explainedVariances.push_back(explainedVariance);
    }

    // Save explained variances to a CSV file
    std::ofstream csvFile(outputCsvFile);

    if (!csvFile.is_open()) {
        throw std::runtime_error("Could not open file to save explained variances.");
    }

    csvFile << "Components,ExplainedVariance\n";
    for (size_t i = 0; i < explainedVariances.size(); ++i) {
        csvFile << (i + 1) << "," << explainedVariances[i] << "\n";
    }
    csvFile.close();
    std::cout << "###  Saved " << outputCsvFile << "   ###" << std::endl;
}