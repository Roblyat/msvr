#include "PCA.h"

void PCA::fit(const cv::Mat &data, int numComponents)
{
    // Ensure the data type is CV_32F
    data.convertTo(data, CV_32F);
    // Fit PCA on the training data
    pca = cv::PCA(data, cv::Mat(), cv::PCA::DATA_AS_ROW, numComponents);
    std::cout << "PCA fitted with " << numComponents << " components." << std::endl;
}

cv::Mat PCA::transform(const cv::Mat &data, const std::string &dataType)
{
    // Ensure the data type is CV_32F
    data.convertTo(data, CV_32F);

    // Transform the data using the fitted PCA model
    cv::Mat projectedData;
    pca.project(data, projectedData);
    std::cout << "TRANSFORMED " << dataType << std::endl;
    std::cout << "Transformed " << dataType << " training data size: " << projectedData.size() << std::endl;
    return projectedData;
}

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
