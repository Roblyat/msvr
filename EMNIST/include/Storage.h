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
    void shuffleData();
    void splitData();

    cv::Mat features;
    cv::Mat targets;

public:
    Storage();
    ~Storage() = default;

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
};