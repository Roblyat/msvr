// SIFT.cpp
#include "SIFT.h"

SIFT::SIFT()
{
    updateSift();
}

void SIFT::updateSift()
{
    // Convert int parameters to their appropriate float equivalents
     sift = cv::SIFT::create(nfeatures, nOctaveLayers, (float)contrastThreshold / 100.0f, (float)edgeThreshold / 100.0f, (float)sigma / 100.0f);
}

void SIFT::onTrackbar(int, void *userdata)
{
    SIFT *ptr = reinterpret_cast<SIFT *>(userdata);
    if (ptr)
    {
        ptr->updateSift();
    }
};

void SIFT::siftTrackbars(std::string window)
{
    cv::namedWindow(window, cv::WINDOW_NORMAL);

    // Create trackbars
    cv::createTrackbar("nFeatures", window, &nfeatures, 400, onTrackbar, this);
    cv::createTrackbar("nOctaveLayers", window, &nOctaveLayers, 8, onTrackbar, this);
    cv::createTrackbar("Contrast Threshold", window, &contrastThreshold, 100, onTrackbar, this);
    cv::createTrackbar("Edge Threshold", window, &edgeThreshold, 100, onTrackbar, this);
    cv::createTrackbar("Sigma", window, &sigma, 100, onTrackbar, this);
};

void SIFT::bfmTrackbars(std::string window)
{
    cv::namedWindow(window, cv::WINDOW_NORMAL);

    // Create trackbars
    cv::createTrackbar("matchThreshold", window, &matchThreshold, 400, onTrackbar, this);
};

void SIFT::siftExtract(cv::Mat &image, cv::Mat &image_with_keypoints, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, bool saveDescriptors)
{
    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    descriptors.convertTo(descriptors, CV_32F);

    if (saveDescriptors) {
        saveCSV();
    }

    cv::drawKeypoints(image, keypoints, image_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imwrite("/home/fhtw_user/msvr/pose_estimation/sift_features.jpg", image_with_keypoints);

}

void SIFT::saveCSV()
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

void SIFT::drawMatches(cv::Mat img_1, std::vector<cv::KeyPoint> keyPoints_1, cv::Mat img_2, std::vector<cv::KeyPoint> keyPoints_2,
                       std::vector<cv::DMatch> matches_1_2, cv::Mat &img_out)
{
    goodMatches.clear();
    for (const auto &match : matches_1_2)
    {
        if (match.distance < (float)matchThreshold)
        {
            goodMatches.push_back(match);
        }
    }

    std::cout << "Good matches: " << goodMatches.size() << std::endl;
    cv::drawMatches(img_1, keyPoints_1, img_2, keyPoints_2, goodMatches, img_out, 
        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}

void SIFT::matchDescriptors(cv::Mat &descriptors)
{   
    std::cout << "Train Descriptors" << descriptors.size() << std::endl;
    std::cout << "Camera Descriptors" << cameraDescriptors.size() << std::endl;

    cv::Mat descriptors_nI = descriptors.colRange(0, descriptors.cols);
    cv::Mat cameraDescriptors_nI = cameraDescriptors.colRange(0, cameraDescriptors.cols);

    // Initialize the matcher with the correct norm type
    cv::BFMatcher matcher(cv::NORM_L2);
    matcher.match(descriptors, cameraDescriptors, matches);

    // Sort the matches based on distance
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
        return a.distance < b.distance;
    });

    std::cout << "Matches: " << matches.size() << std::endl;
}

void SIFT::safeActiveSet(cv::Mat &descriptors)
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

    // std::cout << "Total matches: " << matches.size() << ", Good matches: " << goodMatches.size() << std::endl;
    // goodMatches.clear();
}










































































































void SIFT::showKeyNumbers(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img_with_keypoints, size_t keypointIndex)
{
    if (keypointIndex < keypoints.size())
    {
        const auto &kp = keypoints[keypointIndex];
        cv::putText(img_with_keypoints, std::to_string(keypointIndex), kp.pt, cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(0, 255, 0), 1);
    }
}

    //###########   BF MATCHETR ÜBERGEBEN UND FUNKTION NOCH MEHR AUFTEILEN!!  ##########
void SIFT::useHandpickedKeypoints(const cv::Mat &image, cv::Mat &img_with_handpickedKeypoints, std::vector<cv::KeyPoint> &storePickedKP)
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
        
        // Ensure cameraDescriptors are also of type CV_32F
        if (cameraDescriptors.type() != CV_32F)
        {
            cameraDescriptors.convertTo(cameraDescriptors, CV_32F);
        }

        std::cout << "picked keypoints: " << handpickedKeypoints.size() << std::endl;
        std::cout << "camera keypoints: " << cameraKeypoints.size() << std::endl;
        std::cout << "picked descriptors: " << handpickedDescriptors.size() << std::endl;
        std::cout << "camera descriptors: " << cameraDescriptors.size() << std::endl;

        // matcher.match(handpickedDescriptors, cameraDescriptors, newMatches);

        // drawMatches();
}