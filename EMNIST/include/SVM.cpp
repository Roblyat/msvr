#include "SVM.h"

SVM::SVM()
{
    // Initialize the SVM with default parameters
    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::NU_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
}

void SVM::train(const cv::Mat& features, const cv::Mat& lables) 
{
    svm->setNu(bestNu);
    svm->setGamma(bestGamma);
    this->trainData = cv::ml::TrainData::create(features, cv::ml::ROW_SAMPLE, lables);
    // Train the SVM
    svm->train(trainData, cv::ml::ROW_SAMPLE);
}

float SVM::evaluate(const cv::Mat& testData, const cv::Mat& testLabels)
{
    // Ensure the data type is CV_32F
    cv::Mat floatTestData;
    testData.convertTo(floatTestData, CV_32F);

    // Ensure the labels are in CV_32S for comparison
    cv::Mat intTestLabels;
    testLabels.convertTo(intTestLabels, CV_32S);

    // Evaluate the SVM on the test data
    cv::Mat predictions;
    svm->predict(floatTestData, predictions);

    // Ensure predictions are in CV_32S for comparison
    cv::Mat intPredictions;
    predictions.convertTo(intPredictions, CV_32S);

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < intTestLabels.rows; ++i)
    {
        int predictedLabel = intPredictions.at<int>(i, 0);
        int actualLabel = intTestLabels.at<int>(i, 0);

        if (predictedLabel == actualLabel)
        {
            correct++;
        }
    }
    return static_cast<float>(correct) / intTestLabels.rows;
}

void SVM::optimizeParameters(const cv::Mat& trainData, const cv::Mat& trainLabels, const cv::Mat& valData, const cv::Mat& valLabels)
{
    std::vector<float> nuValues = {0.01, 0.1, 0.2, 0.3};
    std::vector<float> gammaValues = {0.0001, 0.001, 0.01, 0.1};

    for (double nu = 0.01; nu <= 0.5; nu += 0.01)
    {
        for (double gamma = 0.0001; gamma <= 0.001; gamma += 0.0001)
        {
            svm->setNu(nu);
            svm->setGamma(gamma);

            train(trainData, trainLabels);

            float accuracy = evaluate(valData, valLabels);
            std::cout << "Nu: " << nu << ", Gamma: " << gamma << ", Accuracy: " << accuracy * 100 << "%" << std::endl;

            if (accuracy > bestAccuracy)
            {
                bestNu = nu;
                bestGamma = gamma;
                bestAccuracy = accuracy;
            }
        }
    }

    svm->setNu(bestNu);
    svm->setGamma(bestGamma);
    std::cout << "Best Nu: " << bestNu << ", Best Gamma: " << bestGamma << ", Best Accuracy: " << bestAccuracy * 100 << "%" << std::endl;
}
