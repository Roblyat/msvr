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
        cv::Mat targets;

        struct origin
        {
            cv::Mat features;
        }origin;

        struct transformed
        {
            cv::Mat features;

            struct trainSubset
            {
                cv::Mat features;
                cv::Mat targets;
            }trainSubset;

            struct validateSubset
            {
                cv::Mat features;
                cv::Mat targets;
            }validateSubset;

        }transformed;
    };

    struct TestData
    {
        cv::Mat targets;
        
        struct origin
        {
            cv::Mat features;
        }origin;

        struct transformed
        {
            cv::Mat features;
        }transformed;
    };

public:
    Storage();
    ~Storage() = default;

    void convertData();
    void splitValidation();

    TrainData trainData;
    TestData testData;
};