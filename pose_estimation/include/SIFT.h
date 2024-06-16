// classSift.h
#pragma once
#include "../include/classCVBasic.h"

class SIFT : public CVBasic
{

public:
    // constructor
    SIFT();
    // destructor
    ~SIFT(){};

    void updateSift();

    static void onTrackbar(int, void *userdata);

    void siftTrackbars(std::string window);

    void siftExtract(const cv::Mat &image, cv::Mat &img_with_keypoints, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, bool saveDescriptors);

    void bfmTrackbars(std::string window);

    void drawMatches();

    void useHandpickedKeypoints(const cv::Mat &image, cv::Mat &img_with_handpickedKeypoints, std::vector<cv::KeyPoint> &storePickedKP);

    void safeActiveSet();
    
    void matchDescriptors();

    void showKeyNumbers(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img_with_keypoints, size_t keypointIndex);
  

private:
    cv::Ptr<cv::SiftFeatureDetector> sift;
    cv::BFMatcher matcher;
    int nfeatures = 400, nOctaveLayers = 4, contrastThreshold = 10, edgeThreshold = 20, sigma = 55;
    int matchThreshold = 72;
};