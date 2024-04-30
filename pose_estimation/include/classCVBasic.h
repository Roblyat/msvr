// classCVBasic.h
#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

class CVBasic
{

public:
    // constructor
    CVBasic() {};
    // destructor
    ~CVBasic(){};

    // methods
    // load image & use clone to process
    int loadImage(cv::Mat &storage_image)
    {
        storage_image = cv::imread(imgPath).clone();

        if (storage_image.empty())
            std::cerr << "loading image failed." << std::endl;

        return 0;
    };

    int grayScale(const cv::Mat &image, cv::Mat &gray_scaled_image)
    {   
        // scale BRG to GRAY
        cv::cvtColor(image, gray_scaled_image, cv::COLOR_BGR2GRAY);

        return 0;
    };

private:
    // image path
    const std::string imgPath = "/home/fhtw_user/msvr/pose_estimation/bart.png";
};