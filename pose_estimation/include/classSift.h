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
                // Writing the header line with descriptor column names
                file << "Index"; // Include an index header
                for (int j = 0; j < descriptors.cols; ++j)
                {
                    file << ",D" << j; // Append each descriptor header like D0, D1, ..., D127
                }
                file << "\n"; // End the header line

                // Writing the descriptor data
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
                          cv::Mat &camera_img_with_keypoints, cv::Mat &img_matches, cv::Mat &img_with_handpickedKeypoints, 
                          std::vector<cv::DMatch> newMatches, std::vector<cv::KeyPoint> &storePickedKP,
                          size_t oldSize, bool safeThreshold, bool variateThreshold, bool useHandpicked, bool showAllMatches)
    {
        if (variateThreshold)
            cv::createTrackbar("matchThreshold", "match threshold", &matchThreshold, 400, onTrackbar, this);

        std::vector<cv::DMatch> matches;
        cv::BFMatcher matcher(cv::NORM_L2);
        if (!useHandpicked)
            matcher.match(descriptors, cameraDescriptors, matches);

        if (useHandpicked)
        {
            std::ifstream file("/home/fhtw_user/msvr/pose_estimation/activeSet.csv");
            std::vector<std::vector<float>> data;
            std::string line;
            bool isFirstLine = true; // Flag to skip the first line (header)

            // Skip the header line
            std::getline(file, line);

            while (std::getline(file, line))
            {
                if (isFirstLine)
                {
                    isFirstLine = false; // Change the flag after skipping the first line
                    continue;
                }
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

            cv::Mat handpickedDescriptors(data.size(), data[0].size() - 8, CV_32F); // Exclude the first 8 columns (index and keypoint data)
            std::vector<cv::KeyPoint> handpickedKeypoints(data.size());

            for (size_t i = 0; i < data.size(); i++)
            {
                // Load the keypoint data
                cv::KeyPoint kp;
                kp.pt.x = data[i][1];                       // X
                kp.pt.y = data[i][2];                       // Y
                kp.size = data[i][3];                       // Size
                kp.angle = data[i][4];                      // Angle
                kp.response = data[i][5];                   // Response
                kp.octave = static_cast<int>(data[i][6]);   // Octave
                kp.class_id = static_cast<int>(data[i][7]); // ClassID

                handpickedKeypoints[i] = kp;

                // Load the descriptor data
                for (size_t j = 8; j < data[0].size(); j++) // Start from 8 to skip index and keypoint columns
                {
                    handpickedDescriptors.at<float>(i, j - 8) = data[i][j];
                }
            }
            storePickedKP = handpickedKeypoints;
            // Ensure cameraDescriptors are also of type CV_32F
            if (cameraDescriptors.type() != CV_32F)
            {
                cameraDescriptors.convertTo(cameraDescriptors, CV_32F);
            }

            std::cout << "picked keypoints: " << handpickedKeypoints.size() << std::endl;
            std::cout << "camera keypoints: " << cameraKeypoints.size() << std::endl;
            std::cout << "picked descriptors: " << handpickedDescriptors.size() << std::endl;
            std::cout << "camera descriptors: " << cameraDescriptors.size() << std::endl;

            matcher.match(handpickedDescriptors, cameraDescriptors, newMatches);

            cv::drawMatches(img_with_keypoints, handpickedKeypoints, camera_img_with_keypoints, cameraKeypoints, newMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                            std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        }

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
            std::ofstream outFile("/home/fhtw_user/msvr/pose_estimation/activeSet.csv");

            // Write header row
            outFile << "Index,X,Y,Size,Angle,Response,Octave,ClassID";
            for (int i = 0; i < 128; i++)
            { // Assuming you have 128 descriptors
                outFile << ",D" << i;
            }
            outFile << "\n";

            for (const auto &match : goodMatches)
            {
                int queryIdx = match.queryIdx; // Index of the descriptor in the training set
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