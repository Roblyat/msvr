#pragma once
#include "opencv2/opencv.hpp"
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>

class Storage {
public:
    Storage() = default;//: dataReady(false) {}
    ~Storage() = default;

    // Images and keypoints storage
    cv::Mat image;
    cv::Mat gray_scaled_image;
    cv::Mat img_with_keypoints;
    cv::Mat undistortImage;
    std::vector<cv::KeyPoint> keypoints;

    cv::Mat cameraImage;
    cv::Mat undistortCameraImage;
    cv::Mat camera_img_with_keypoints;
    std::vector<cv::KeyPoint> cameraKeypoints;
};
