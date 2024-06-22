#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

//###############   class CVBasic   ###############
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


//###############   class SIFT  ###############
class SIFT : public CVBasic
{

public:
    // Constructor
    SIFT();

    // Destructor
    ~SIFT(){};

    // Updates the SIFT detector with current parameter values
    void updateSift();

    // Static callback function for trackbar events
    static void onTrackbar(int, void *userdata);

    // Creates trackbars for SIFT parameters
    void siftTrackbars(std::string window);

    // Extracts SIFT features from the image and optionally saves the descriptors to a CSV file
    void siftExtract(bool saveDescriptors);

    // Saves descriptors to a CSV file
    void saveCSV();

    // Creates trackbars for BFMatcher parameters
    void bfmTrackbars(std::string window);

    // Draws matches between keypoints from two images and stores the result in img_out
    void drawMatches(cv::Mat img_1, std::vector<cv::KeyPoint> keyPoints_1, cv::Mat img_2, std::vector<cv::KeyPoint> keyPoints_2,
                     std::vector<cv::DMatch> matches_1_2, cv::Mat &img_out);

    // Saves the active set of descriptors to a CSV file
    void safeActiveSet(cv::Mat &descriptors);

    // Matches descriptors using BFMatcher
    void matchDescriptors(cv::Mat &descriptors);

    // Shows the keypoint numbers on the image with keypoints
    void showKeyNumbers(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img_with_keypoints, size_t keypointIndex);

private:
    cv::Ptr<cv::SiftFeatureDetector> sift; // SIFT feature detector
    int nfeatures = 400; // Number of features
    int nOctaveLayers = 4; // Number of octave layers
    int contrastThreshold = 10; // Contrast threshold
    int edgeThreshold = 20; // Edge threshold
    int sigma = 55; // Sigma value
    int matchThreshold = 100; // Match threshold
};

// Constructor: Initializes the SIFT detector with current parameters
SIFT::SIFT()
{
    updateSift();
}

// Updates the SIFT detector with current parameter values
void SIFT::updateSift()
{
    // Convert int parameters to their appropriate float equivalents and create the SIFT detector
    sift = cv::SIFT::create(nfeatures, nOctaveLayers, (float)contrastThreshold / 100.0f, (float)edgeThreshold / 100.0f, (float)sigma / 100.0f);
}

// Static callback function for trackbar events
void SIFT::onTrackbar(int, void *userdata)
{
    SIFT *ptr = reinterpret_cast<SIFT *>(userdata);
    if (ptr)
    {
        ptr->updateSift();
    }
}

// Creates trackbars for SIFT parameters
void SIFT::siftTrackbars(std::string window)
{
    cv::namedWindow(window, cv::WINDOW_NORMAL);

    // Create trackbars
    cv::createTrackbar("nFeatures", window, &nfeatures, 400, onTrackbar, this);
    cv::createTrackbar("nOctaveLayers", window, &nOctaveLayers, 8, onTrackbar, this);
    cv::createTrackbar("Contrast Threshold", window, &contrastThreshold, 100, onTrackbar, this);
    cv::createTrackbar("Edge Threshold", window, &edgeThreshold, 100, onTrackbar, this);
    cv::createTrackbar("Sigma", window, &sigma, 100, onTrackbar, this);
}

// Creates trackbars for BFMatcher parameters
void SIFT::bfmTrackbars(std::string window)
{
    cv::namedWindow(window, cv::WINDOW_NORMAL);

    // Create trackbar for match threshold
    cv::createTrackbar("matchThreshold", window, &matchThreshold, 400, onTrackbar, this);
}

// Extracts SIFT features from the image and optionally saves the descriptors to a CSV file
void SIFT::siftExtract(bool saveDescriptors)
{
    // Detect and compute keypoints and descriptors
    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    // Convert descriptors to CV_32F format
    descriptors.convertTo(descriptors, CV_32F);

    // Save descriptors to CSV file if required
    if (saveDescriptors) {
        saveCSV();
    }

    // Draw keypoints on the image and save the result
    cv::drawKeypoints(image, keypoints, img_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imwrite("/home/fhtw_user/msvr/pose_estimation/sift_features.jpg", img_with_keypoints);
}

// Saves descriptors to a CSV file
void SIFT::saveCSV()
{
    std::ofstream file("/home/fhtw_user/msvr/pose_estimation/exam/descriptors.csv");
    if (file.is_open())
    {
        // Write header line with descriptor column names
        file << "Index"; // Include an index header
        for (int j = 0; j < descriptors.cols; ++j)
        {
            file << ",D" << j; // Append each descriptor header like D0, D1, ..., D127
        }
        file << "\n"; // End the header line

        // Write descriptor data
        for (int i = 0; i < descriptors.rows; ++i)
        {
            file << i; // Start with the row index
            for (int j = 0; j < descriptors.cols; ++j)
            {
                file << "," << descriptors.at<float>(i, j); // Append descriptor values separated by commas
            }
            file << "\n"; // End the line for each descriptor row
        }
        file.close();
        std::cout << "Descriptors saved in 'descriptors.csv'" << std::endl;
    }
    else
    {
        std::cerr << "Failed saving descriptors" << std::endl;
    }
}

// Draws matches between keypoints from two images and stores the result in img_out
void SIFT::drawMatches(cv::Mat img_1, std::vector<cv::KeyPoint> keyPoints_1, cv::Mat img_2, std::vector<cv::KeyPoint> keyPoints_2,
                       std::vector<cv::DMatch> matches_1_2, cv::Mat &img_out)
{
    // Clear previous good matches
    goodMatches.clear();
    
    // Filter matches based on distance
    for (const auto &match : matches_1_2)
    {
        if (match.distance < (float)matchThreshold)
        {
            goodMatches.push_back(match);
        }
    }

    std::cout << "Good matches: " << goodMatches.size() << std::endl;
    
    // Draw good matches
    cv::drawMatches(img_1, keyPoints_1, img_2, keyPoints_2, goodMatches, img_out, 
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}

// Matches descriptors using BFMatcher
void SIFT::matchDescriptors(cv::Mat &trainDescriptors)
{
    std::cout << "Train Descriptors: " << trainDescriptors.size() << std::endl;
    std::cout << "Camera Descriptors: " << descriptors.size() << std::endl;

    cv::Mat trainDescriptors_nI = trainDescriptors.colRange(0, trainDescriptors.cols);
    cv::Mat descriptors_nI = descriptors.colRange(0, descriptors.cols);

    // Initialize the matcher with the correct norm type
    cv::BFMatcher matcher(cv::NORM_L2);
    matcher.match(trainDescriptors, descriptors, matches);

    // Sort the matches based on distance
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
        return a.distance < b.distance;
    });

    std::cout << "Matches: " << matches.size() << std::endl;
}

// Saves the active set of descriptors to a CSV file
void SIFT::safeActiveSet(cv::Mat &trainDescriptors)
{
    std::ofstream outFile("/home/fhtw_user/msvr/pose_estimation/exam/activeSet.csv");

    // Write header row
    outFile << "Index,X,Y,Size,Angle,Response,Octave,ClassID";
    for (int i = 0; i < 128; i++)
    {
        outFile << ",D" << i;
    }
    outFile << "\n";

    // Write data for good matches
    for (const auto &match : goodMatches)
    {
        size_t queryIdx = match.queryIdx; // Index of the descriptor in the training set
        if (queryIdx < descriptors.rows && queryIdx < keypoints.size())
        {
            const cv::KeyPoint &kp = keypoints[queryIdx];
            const cv::Mat descriptor = descriptors.row(queryIdx);

            // Write keypoint data
            outFile << queryIdx << "," << kp.pt.x << "," << kp.pt.y << "," << kp.size << ","
                    << kp.angle << "," << kp.response << "," << kp.octave << "," << kp.class_id;

            // Write descriptor data
            for (int j = 0; j < descriptor.cols; j++)
            {
                outFile << "," << descriptor.at<float>(j);
            }
            outFile << "\n";
        }
    }
    std::cout << "Saved " << goodMatches.size() << " good match descriptors to 'activeSet.csv'." << std::endl;
    outFile.close();
}


// ##############   main loop  ################
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

