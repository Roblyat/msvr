#include "CVBasic.h"
#include "SIFT.h"

// Global variable to keep track of the current keypoint index
size_t keyPointIndex = 0;

int main()
{   
    SIFT siftTrain, siftCamera;

    // Load the training image
    siftTrain.loadImage();

    // Undistort the training image
    siftTrain.undistort(siftTrain.image, siftTrain.undistortImage);

    // Extract SIFT features from the training image and save descriptors
    siftTrain.siftExtract(true);

    // Open a video file for processing
    cv::VideoCapture cap("/home/fhtw_user/msvr/pose_estimation/exam/video2Mp4.mp4"); // Change the path if necessary

    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }

    char key = 0;

    while (key != 'd')
    {   
        // Create trackbars for BFMatcher parameters
        siftCamera.bfmTrackbars("match threshold");

        // Capture the next frame from the video
        cap >> siftCamera.image;
        if (siftCamera.image.rows <= 0)
        {
            std::cout << "Cannot use camera stream\n";
            return -1;
        }

        // Undistort the captured frame
        siftCamera.undistort(siftCamera.image, siftCamera.undistortImage);

        // Extract SIFT features from the captured frame without saving descriptors
        siftCamera.siftExtract(false);

        // Display keypoints on the training and camera images (commented out)
        // cv::imshow("Input keyPoints", siftTrain.img_with_keypoints);
        // cv::imshow("Camera keyPoints", siftCamera.camera_img_with_keypoints);

        // Match descriptors between the training image and the current frame
        siftCamera.matchDescriptors(siftTrain.descriptors);

        // Draw matches between the training image and the current frame
        siftCamera.drawMatches(siftTrain.img_with_keypoints, siftTrain.keypoints, siftCamera.img_with_keypoints, siftCamera.keypoints, siftCamera.matches, siftCamera.img_matches);
        
        // Save the active set of descriptors
        siftCamera.safeActiveSet(siftTrain.descriptors);

        // Display the matches
        cv::imshow("Matches", siftCamera.img_matches);

        // Wait for a key press
        key = cv::waitKey(1);
    }

    return 0;
};
