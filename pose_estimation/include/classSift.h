// classSift.h
#pragma once
#include "../include/classCVBasic.h"

class SIFT : public CVBasic
{

public:
    // constructor
    SIFT()
    {
        updateSift();
    };
    // destructor
    ~SIFT(){};

    void updateSift()
    {
        // Convert int parameters to their appropriate float equivalents
        sift = cv::SiftFeatureDetector::create(nfeatures, nOctaveLayers, (float)contrastThreshold / 100.0f, (float)edgeThreshold / 100.0f,
                                               (float)sigma / 100.0f);
    };

    static void onTrackbar(int, void *userdata)
    {
        SIFT *ptr = reinterpret_cast<SIFT *>(userdata);
        if (ptr)
        {
            ptr->updateSift();
        }
    };

    void trackbars()
    {
        cv::namedWindow("SIFT Parameters", cv::WINDOW_NORMAL);

        // Create trackbars
        cv::createTrackbar("nFeatures", "SIFT Parameters", &nfeatures, 400, onTrackbar, this);
        cv::createTrackbar("nOctaveLayers", "SIFT Parameters", &nOctaveLayers, 8, onTrackbar, this);
        cv::createTrackbar("Contrast Threshold", "SIFT Parameters", &contrastThreshold, 100, onTrackbar, this);
        cv::createTrackbar("Edge Threshold", "SIFT Parameters", &edgeThreshold, 100, onTrackbar, this);
        cv::createTrackbar("Sigma", "SIFT Parameters", &sigma, 100, onTrackbar, this);
    };

    int siftExtract(const cv::Mat &image, cv::Mat &img_with_keypoints, std::vector<cv::KeyPoint> &storageKeypoints)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        // safe descriptors in csv file
        std::ofstream file("/home/fhtw_user/msvr/pose_estimation/descriptors.csv");
        if (file.is_open())
        {
            for (int i = 0; i < descriptors.rows; ++i)
            {
                for (int j = 0; j < descriptors.cols; ++j)
                {
                    file << descriptors.at<float>(i, j);
                    if (j < descriptors.cols - 1)
                        file << ",";
                }
                file << "\n";
            }
            file.close();
            std::cout << "descriptors saved in 'descriptors.csv'" << std::endl;
        }
        else
        {
            std::cerr << "failed saveing descriptors" << std::endl;
        }

        // safe image with keypoints
        cv::drawKeypoints(image, keypoints, img_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        cv::imwrite("/home/fhtw_user/msvr/pose_estimation/sift_features.jpg", img_with_keypoints);

        storageKeypoints = keypoints;

        return 0;
    };

    int showKeyNumbers(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img_with_keypoints, size_t keypointIndex)
    {
        if (keypointIndex < keypoints.size()) {
            const auto &kp = keypoints[keypointIndex];
            cv::putText(img_with_keypoints, std::to_string(keypointIndex), kp.pt, cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(0, 255, 0), 1);
        }

        return 0;
    }

private:
    cv::Ptr<cv::SiftFeatureDetector> sift;
    int nfeatures = 400, nOctaveLayers = 5, contrastThreshold = 8, edgeThreshold = 20, sigma = 55;
};
