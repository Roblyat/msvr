// main.cpp
#include "CVBasic.h"
#include "SIFT.h"

size_t keyPointIndex = 0;

int main()
{   
    SIFT siftTrain, siftCamera;
    siftTrain.loadImage();

    siftTrain.undistort(siftTrain.image, siftTrain.undistortImage);

    siftTrain.siftExtract(siftTrain.image, siftTrain.img_with_keypoints, siftTrain.keypoints, siftTrain.descriptors, true);


    // cv::namedWindow("Camera Features", cv::WINDOW_NORMAL);
    // cv::namedWindow("match threshold", cv::WINDOW_NORMAL);

    char key = 0;
    cv::VideoCapture cap("/home/fhtw_user/msvr/pose_estimation/webcam/video2Mp4.mp4"); //"/home/fhtw_user/msvr/pose_estimation/webcam/videoMp4.mp4"

    while (key != 'd')
    {   
    
        // siftTrain.trackbars("train parameters");
        siftCamera.bfmTrackbars("match threshold");

        cap >> siftCamera.cameraImage;
        if (siftCamera.cameraImage.rows <= 0)
        {
            std::cout << "Cannot use camera stream\n";
            return (-1);
        }

        siftCamera.undistort(siftCamera.cameraImage, siftCamera.undistortCameraImage);

        siftCamera.siftExtract(siftCamera.undistortCameraImage, siftCamera.camera_img_with_keypoints, siftCamera.cameraKeypoints, siftCamera.cameraDescriptors, false);
       
        // cv::imshow("Input keyPoints", siftTrain.img_with_keypoints);
        // cv::imshow("Camera keyPoints", siftCamera.camera_img_with_keypoints);

        siftCamera.matchDescriptors(siftTrain.descriptors);

        siftCamera.drawMatches(siftTrain.img_with_keypoints, siftTrain.keypoints, siftCamera.camera_img_with_keypoints, siftCamera.cameraKeypoints, siftCamera.matches, siftCamera.img_matches);
        
        siftCamera.safeActiveSet(siftTrain.descriptors);

        cv::imshow("Matches", siftCamera.img_matches);

        key = cv::waitKey(1);
    };

    return 0;
};