#pragma once
#include "CVBasic.h"

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
