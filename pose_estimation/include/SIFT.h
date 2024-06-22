// classSift.h
#pragma once
#include "CVBasic.h"

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

    void siftExtract(cv::Mat &image, cv::Mat &image_with_keypoints, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, bool saveDescriptors);

    void saveCSV();

    void bfmTrackbars(std::string window);

    void drawMatches(cv::Mat Img_1, std::vector<cv::KeyPoint> keyPoints_1, cv::Mat img_2, std::vector<cv::KeyPoint> keyPoints_2,
                            std::vector<cv::DMatch> matches_1_2, cv::Mat Img_out);

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