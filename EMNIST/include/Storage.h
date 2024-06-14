// Storage.h
#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>
#include <random>

class Storage
{
private:
    void loadData();
    void extractRowsForLetters();
    void printRows(cv::Mat targets) const;
    void printMinMax(const cv::Mat& mat) const;
    void shuffleData();
    void splitData();
    void standardizeData();

    cv::Mat features;
    cv::Mat targets;

    struct TrainData
    {
        cv::Mat features;
        cv::Mat targets;
    };

    struct TestData
    {
        cv::Mat features;
        cv::Mat targets;
    };

public:
    Storage();
    ~Storage() = default;

    TrainData trainData;
    TestData testData;
};