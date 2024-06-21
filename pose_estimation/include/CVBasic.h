// classCVBasic.h
#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include "../include/classStorage.h"

class CVBasic : public Storage
{

public:
    // constructor
    CVBasic() {};
    // destructor
    ~CVBasic() {};

    // methods
    // load image & use clone to process
    int loadImage(cv::Mat &storage_image);

    int grayScale(const cv::Mat &image, cv::Mat &gray_scaled_image);

    void undistort(cv::Mat &image, cv::Mat &undistortImage);

private:
    // image path
    const std::string imgPath = "/home/fhtw_user/msvr/pose_estimation/webcam/tp6.jpg";

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 913.086760, 0.000000, 624.176299,
                            0.000000, 907.672223, 394.805003,
                            0.000000, 0.000000, 1.000000);

    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.119547, -0.187557, 0.000381, -0.000114, 0.000000);
};