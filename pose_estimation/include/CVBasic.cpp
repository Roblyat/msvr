#include "CVBasic.h"

    // load image & use clone to process
    int CVBasic::loadImage(cv::Mat &storage_image)
    {
        storage_image = cv::imread(imgPath).clone();

        if (storage_image.empty())
            std::cerr << "loading image failed." << std::endl;

        return 0;
    };

    int CVBasic::grayScale(const cv::Mat &image, cv::Mat &gray_scaled_image)
    {
        // scale BRG to GRAY
        cv::cvtColor(image, gray_scaled_image, cv::COLOR_BGR2GRAY);

        return 0;
    };

    void CVBasic::undistort(cv::Mat &image, cv::Mat &undistortImage)
    {
        cv::undistort(image, undistortImage, cameraMatrix, distCoeffs);
    };
