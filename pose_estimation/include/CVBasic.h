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
    ~CVBasic() {};

    // methods
    // load image & use clone to process
    int loadImage();

    int grayScale(const cv::Mat &image, cv::Mat &gray_scaled_image);

    void undistort(cv::Mat &image, cv::Mat &undistortImage);

    // Images and keypoints storage
    cv::Mat image;
    cv::Mat gray_scaled_image;
    cv::Mat img_with_keypoints;
    cv::Mat img_with_handpickedKeypoints;
    cv::Mat undistortImage;
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;

    cv::Mat cameraImage;
    cv::Mat undistortCameraImage;
    cv::Mat camera_img_with_keypoints;
    cv::Mat cameraDescriptors;
    std::vector<cv::KeyPoint> cameraKeypoints;
    std::vector<cv::KeyPoint> storePickedKP;

    std::vector<cv::DMatch> goodMatches;
    std::vector<cv::DMatch> matches;
    size_t oldSize;
    cv::Mat img_matches;

private:
    // image path
    const std::string imgPath = "/home/fhtw_user/msvr/pose_estimation/webcam/tp6.jpg";

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 913.086760, 0.000000, 624.176299,
                            0.000000, 907.672223, 394.805003,
                            0.000000, 0.000000, 1.000000);

    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.119547, -0.187557, 0.000381, -0.000114, 0.000000);
};