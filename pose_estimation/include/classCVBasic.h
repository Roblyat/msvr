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

    void undistort(cv::Mat &image, cv::Mat &undistortImage)
    {
        cv::undistort(image, undistortImage, cameraMatrix, distCoeffs);
    };

    void startCamera(cv::Mat &cameraImage)
    {
        cv::VideoCapture cap(0); // Open the default camera
        if (!cap.isOpened())
        {
            std::cerr << "Error opening video stream or file" << std::endl;
            return;
        }

        while (true)
        {
            cap >> cameraImage; // Capture a new image frame
            if (cameraImage.empty())
            {
                std::cerr << "Received empty frame. Exiting..." << std::endl;
                break; // Exit if the frame is empty
            }

            cv::imshow("Camera Image", cameraImage); // Display the frame
            if (cv::waitKey(1) == 'q')
            {
                break; // Exit loop on 'q' key press
            }
        }

        cap.release(); // Release the video capture object
    }

private:
    // image path
    const std::string imgPath = "/home/fhtw_user/msvr/pose_estimation/webcam/tp6.jpg";

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 913.086760, 0.000000, 624.176299,
                            0.000000, 907.672223, 394.805003,
                            0.000000, 0.000000, 1.000000);

    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.119547, -0.187557, 0.000381, -0.000114, 0.000000);
};