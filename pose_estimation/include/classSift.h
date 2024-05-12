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

    void trackbars(std::string window)
    {
        cv::namedWindow(window, cv::WINDOW_NORMAL);

        // Create trackbars
        cv::createTrackbar("nFeatures", window, &nfeatures, 400, onTrackbar, this);
        cv::createTrackbar("nOctaveLayers", window, &nOctaveLayers, 8, onTrackbar, this);
        cv::createTrackbar("Contrast Threshold", window, &contrastThreshold, 100, onTrackbar, this);
        cv::createTrackbar("Edge Threshold", window, &edgeThreshold, 100, onTrackbar, this);
        cv::createTrackbar("Sigma", window, &sigma, 100, onTrackbar, this);
    };

    int siftExtract(const cv::Mat &image, cv::Mat &img_with_keypoints, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, bool saveDescriptors)
    {
        sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        // safe descriptors in csv file
        if (saveDescriptors)
        {
            std::ofstream file("/home/fhtw_user/msvr/pose_estimation/descriptors.csv");
            if (file.is_open())
            {
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

        // safe image with keypoints
        cv::drawKeypoints(image, keypoints, img_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        cv::imwrite("/home/fhtw_user/msvr/pose_estimation/sift_features.jpg", img_with_keypoints);

        return 0;
    };

    void matchDescriptors(const cv::Mat &descriptors, const cv::Mat &cameraDescriptors, std::vector<cv::DMatch> &goodMatches,
                          std::vector<cv::KeyPoint> &keypoints, std::vector<cv::KeyPoint> &cameraKeypoints, cv::Mat &img_with_keypoints,
                          cv::Mat &camera_img_with_keypoints, cv::Mat &img_matches, size_t oldSize,
                          bool safeThreshold, bool variateThreshold, bool useHandpicked, bool showAllMatches)
    {
        if (variateThreshold)
            cv::createTrackbar("matchThreshold", "match threshold", &matchThreshold, 400, onTrackbar, this);

        std::vector<cv::DMatch> matches;
        cv::BFMatcher matcher(cv::NORM_L2); // Using L2 norm, adjust if needed
        if (!useHandpicked)
            matcher.match(descriptors, cameraDescriptors, matches);

        // use handpicked descriptors
        if (useHandpicked)
        {
            std::ifstream file("/home/fhtw_user/msvr/pose_estimation/threshold.csv");
            std::vector<std::vector<float>> data;
            std::string line;
            while (std::getline(file, line))
            {
                std::stringstream lineStream(line);
                std::string cell;
                std::vector<float> row;

                // Read each cell in the row
                while (std::getline(lineStream, cell, ','))
                {
                    row.push_back(std::stof(cell));
                }

                data.push_back(row);
            }

            cv::Mat handpickedDescriptors(data.size(), data[0].size() - 1, CV_32F); // Exclude the first column which contains indices

            for (size_t i = 0; i < data.size(); i++)
            {
                for (size_t j = 1; j < data[0].size(); j++) // Start from 1 to skip the index
                {
                    handpickedDescriptors.at<float>(i, j - 1) = data[i][j];
                }
            }

            // std::cout << handpickedDescriptors.size() << std::endl;

            matcher.match(handpickedDescriptors, cameraDescriptors, matches);

            cv::drawMatches(img_with_keypoints, keypoints, camera_img_with_keypoints, cameraKeypoints, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                            std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        };

        std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b)
                  { return a.distance < b.distance; });

        if (showAllMatches)
            cv::drawMatches(img_with_keypoints, keypoints, camera_img_with_keypoints, cameraKeypoints, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                            std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        for (const auto &match : matches)
        {
            if (match.distance < (float)matchThreshold)
            { // Filter matches based on the distance. Lower means better.
                goodMatches.push_back(match);
            }
        }

        if (!showAllMatches && variateThreshold)
            cv::drawMatches(img_with_keypoints, keypoints, camera_img_with_keypoints, cameraKeypoints, goodMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                            std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        if (safeThreshold)
        {
            std::ofstream outFile("/home/fhtw_user/msvr/pose_estimation/threshold.csv");
            for (const auto &match : goodMatches)
            {
                int trainIdx = match.trainIdx; // Index of the descriptor in the training set
                if (trainIdx < descriptors.rows)
                {
                    outFile << trainIdx; // First column is the index
                    const cv::Mat descriptor = descriptors.row(trainIdx);
                    for (int j = 0; j < descriptor.cols; j++)
                    {
                        outFile << "," << descriptor.at<float>(j); // Append descriptor values
                    }
                    outFile << "\n";
                }
            }
            std::cout << "Saved " << goodMatches.size() << " good match descriptors to 'threshold.csv'." << std::endl;
            outFile.close();

            std::cout << "Total matches: " << matches.size() << ", Good matches: " << goodMatches.size() << std::endl;
            goodMatches.clear();
        };
    };

    int showKeyNumbers(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img_with_keypoints, size_t keypointIndex)
    {
        if (keypointIndex < keypoints.size())
        {
            const auto &kp = keypoints[keypointIndex];
            cv::putText(img_with_keypoints, std::to_string(keypointIndex), kp.pt, cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(0, 255, 0), 1);
        }

        return 0;
    }

private:
    cv::Ptr<cv::SiftFeatureDetector> sift;
    int nfeatures = 400, nOctaveLayers = 4, contrastThreshold = 10, edgeThreshold = 20, sigma = 55;
    int matchThreshold = 72;
};