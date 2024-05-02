//classStorage
#pragma once
#include "opencv2/opencv.hpp"

struct Storage {  
public:
    //constuctor 
    Storage() = default;
    //destructor
    ~Storage() = default;

    //images
    cv::Mat image;
    cv::Mat gray_scaled_image;
    cv::Mat img_with_keypoints;
    cv::Mat undistortImage;
    std::vector<cv::KeyPoint> keypoints;
    // //vector for channels BGR
    // std::vector<cv::Mat> channelsBGR;
    // //mean kernal
    // cv::Mat meanKernel = cv::Mat::ones(3,3, CV_32F)* 1/9;
    // //gaussian kernal
    // cv::Mat gaussianKernel = (cv::Mat_<float>(3,3) <<  1/16.0, 2/16.0, 1/16.0,
    //                                                    2/16.0, 4/16.0, 2/16.0,
    //                                                    1/16.0, 2/16.0, 1/16.0);
    // //sobel X/Y
    // cv::Mat sobelXKernel = (cv::Mat_<float>(3,3) << -1, 0, 1,
    //                                                 -2, 0, 2,
    //                                                 -1, 0, 1);

    // cv::Mat sobelYKernel = (cv::Mat_<float>(3,3) <<
    // -1, -2, -1,
    //  0,  0,  0,
    //  1,  2,  1);
};
