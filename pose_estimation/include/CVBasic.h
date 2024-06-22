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
    // Constructor
    CVBasic() {};

    // Destructor
    ~CVBasic() {};

    // Methods

    // Loads the image from the specified path and stores it in the 'image' member variable
    int loadImage();

    // Undistorts the input image using the provided camera matrix and distortion coefficients, stores result in 'undistortImage'
    void undistort(cv::Mat &image, cv::Mat &undistortImage);

    // Images and keypoints storage
    cv::Mat image; // Original input image
    cv::Mat img_matches; // Image showing matches between keypoints
    cv::Mat img_with_keypoints; // Image with keypoints drawn on it
    cv::Mat img_with_handpickedKeypoints; // Image with handpicked keypoints drawn on it
    cv::Mat undistortImage; // Undistorted version of the input image
    cv::Mat descriptors; // Descriptors of the keypoints
    std::vector<cv::KeyPoint> keypoints; // Vector storing detected keypoints

    std::vector<cv::DMatch> goodMatches; // Vector storing good matches (filtered based on criteria)
    std::vector<cv::DMatch> matches; // Vector storing all matches

private:
    // Image path
    const std::string imgPath = "/home/fhtw_user/msvr/pose_estimation/exam/tp6.jpg";

    // Camera matrix (intrinsic parameters)
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 913.086760, 0.000000, 624.176299,
                            0.000000, 907.672223, 394.805003,
                            0.000000, 0.000000, 1.000000);

    // Distortion coefficients
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.119547, -0.187557, 0.000381, -0.000114, 0.000000);
};
