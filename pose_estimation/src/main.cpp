// main.cpp
#include "../include/classCVBasic.h"
#include "../include/classStorage.h"
#include "../include/classSift.h"

Storage storage;

int main()
{
    SIFT siftTrain, siftCamera;
    siftTrain.loadImage(storage.image);
    siftTrain.undistort(storage.image, storage.undistortImage);
    siftTrain.siftExtract(storage.undistortImage, storage.img_with_keypoints, storage.keypoints, storage.descriptors, true);

    cv::namedWindow("Camera Features", cv::WINDOW_NORMAL);
    // cv::namedWindow("match threshold", cv::WINDOW_NORMAL);

    char key = 0;
    cv::VideoCapture cap(0);

    while (key != 'd')
    {   
        // siftTrain.trackbars("train parameters");
        // siftCamera.trackbars("camera parameters");
        
        cap >> storage.cameraImage;
        if (storage.cameraImage.rows <= 0)
        {
            std::cout << "Cannot use camera stream\n";
            return (-1);
        }

        siftCamera.undistort(storage.cameraImage, storage.undistortCameraImage);

        siftCamera.siftExtract(storage.undistortCameraImage, storage.camera_img_with_keypoints, storage.cameraKeypoints,
            storage.cameraDescriptors, false);

        siftCamera.matchDescriptors(storage.descriptors, storage.cameraDescriptors,
                                    storage.goodMatches, storage.keypoints, storage.cameraKeypoints, storage.img_with_keypoints,
                                    storage.camera_img_with_keypoints, storage.img_matches, storage.oldSize, false, false, true, false);

        cv::imshow("Camera Features", storage.img_matches);

        key = cv::waitKey(1);
    };

    return 0;
};