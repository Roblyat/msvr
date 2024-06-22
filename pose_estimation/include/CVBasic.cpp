#include "CVBasic.h"

// Loads the image from the specified path and clones it to the 'image' member variable
// Returns 0 upon completion, prints an error message if loading fails
int CVBasic::loadImage()
{
    image = cv::imread(imgPath).clone();

    if (image.empty())
        std::cerr << "loading image failed." << std::endl;

    return 0;
};

// Undistorts the input image using the provided camera matrix and distortion coefficients
// Stores the result in the 'undistortImage' member variable
void CVBasic::undistort(cv::Mat &image, cv::Mat &undistortImage)
{
    cv::undistort(image, undistortImage, cameraMatrix, distCoeffs);
};
