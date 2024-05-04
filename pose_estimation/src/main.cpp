#include "../include/classCVBasic.h"
#include "../include/classStorage.h"
#include "../include/classSift.h"

Storage storage;

int main()
{
    //size_t keypointIndex = 0;

    SIFT siftTrain, siftCamera;
    siftTrain.loadImage(storage.image);
    siftTrain.undistort(storage.image, storage.undistortImage);
    siftTrain.siftExtract(storage.undistortImage, storage.img_with_keypoints, storage.keypoints);

    cv::namedWindow("SIFT Features", cv::WINDOW_NORMAL);
    cv::namedWindow("Camera Image", cv::WINDOW_NORMAL);

    while (true)
    {
        cv::imshow("SIFT Features", storage.img_with_keypoints);

        siftCamera.startCamera(storage.cameraImage);
        cv::imshow("Camera Image", storage.cameraImage);
        //siftCamera.undistort(storage.cameraImage, storage.undistortCameraImage);
        //siftCamera.siftExtract(storage.undistortCameraImage, storage.camera_img_with_keypoints, storage.keypoints);
        //cv::imshow("Camera Image", storage.camera_img_with_keypoints);

        //siftCamera.undistort(storage.cameraImage, storage.undistortCameraImage);
        //siftCamera.siftExtract(storage.undistortCameraImage, storage.camera_img_with_keypoints, storage.keypoints);

        int key = cv::waitKey(1); // Lower the delay to ensure GUI responsiveness
        if (key == 27) break; // Exit on ESC
    };

    return 0;
};


    // Make a fresh copy of the base image each time
    //cv::Mat displayImage = storage.img_with_keypoints.clone();

    // Show features
    //siftTrain.showKeyNumbers(storage.keypoints, displayImage, keypointIndex);
    //cv::imshow("SIFT Features", displayImage);

    // if (keypointIndex < storage.keypoints.size() - 1)
    // {
    //     keypointIndex++; // Increment to show the next keypoint
    // }
    // else
    // {
    //     keypointIndex = 0; // Optionally reset to start over
    // }

    // int key = cv::waitKey(0); // Wait for key press
    // if (key == 27)
    //     break; // Exit on ESC

        