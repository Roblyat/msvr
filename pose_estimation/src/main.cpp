#include "../include/classCVBasic.h"
#include "../include/classStorage.h"
#include "../include/classSift.h"

Storage storage;

int main()
{
    size_t keypointIndex = 0;

    SIFT sift;
    sift.loadImage(storage.image);
    sift.undistort(storage.image, storage.undistortImage);
    sift.siftExtract(storage.undistortImage, storage.img_with_keypoints, storage.keypoints);

    while (true)
    {   
        //trackbars
        // sift.trackbars();
        cv::namedWindow("SIFT Features", cv::WINDOW_NORMAL);

        // Make a fresh copy of the base image each time
        cv::Mat displayImage = storage.img_with_keypoints.clone();

        // Show features
        sift.showKeyNumbers(storage.keypoints, displayImage, keypointIndex);
        cv::imshow("SIFT Features", displayImage);

        int key = cv::waitKey(0); // Wait for key press

        if (key == 27)
            break; // Exit on ESC
        if (keypointIndex < storage.keypoints.size() - 1)
        {
            keypointIndex++; // Increment to show the next keypoint
        }
        else
        {
            keypointIndex = 0; // Optionally reset to start over
        }
    };

    return 0;
}
