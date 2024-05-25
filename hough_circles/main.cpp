#include "iostream"
#include "opencv2/opencv.hpp"

struct Storage
{
    // images
    cv::Mat image;
    cv::Mat convoloutedImage;
    cv::Mat graySegmentedImage;
    cv::Mat sobledImage;
    cv::Mat otsuImage;
    // vector for channels BGR
    std::vector<cv::Mat> channelsBGR;
    // mean kernal
    cv::Mat meanKernel = cv::Mat::ones(3, 3, CV_32F) * 1 / 9;
    // gaussian kernal
    cv::Mat gaussianKernel = (cv::Mat_<float>(3, 3) << 1 / 16.0, 2 / 16.0, 1 / 16.0,
                              2 / 16.0, 4 / 16.0, 2 / 16.0,
                              1 / 16.0, 2 / 16.0, 1 / 16.0);
    // sobel X/Y
    cv::Mat sobelXKernel = (cv::Mat_<float>(3, 3) << -1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1);

    cv::Mat sobelYKernel = (cv::Mat_<float>(3, 3) << -1, -2, -1,
                            0, 0, 0,
                            1, 2, 1);
};

class CvBasic
{
private:
    int lh = 50, uh = 80, ls = 50, us = 130, lv = 160, uv = 200;

public:
    // constructor
    CvBasic()
    {
        // initialize the segmentation window and sliders
        cv::namedWindow("Segmentation");
        cv::createTrackbar("Lower Hue", "Segmentation", &lh, 180);
        cv::createTrackbar("Upper Hue", "Segmentation", &uh, 180);
        cv::createTrackbar("Lower Saturation", "Segmentation", &ls, 255);
        cv::createTrackbar("Upper Saturation", "Segmentation", &us, 255);
        cv::createTrackbar("Lower Value", "Segmentation", &lv, 255);
        cv::createTrackbar("Upper Value", "Segmentation", &uv, 255);
    };
    // destructor
    ~CvBasic(){};

    // methods
    // load image & use clone to process
    cv::Mat loadImage(std::string imgPath)
    {
        cv::Mat image = cv::imread(imgPath).clone();
        return image;
    };
    // show image dimensions
    void showImgDimension(cv::Mat image)
    {
        std::cout << "[" << image.rows << "x" << image.cols << "x" << image.channels() << "]" << std::endl;
    };
    // convert the input BGR image to HSV color space for easier color segmentation
    void segmentImage(const cv::Mat &image, cv::Mat &storageImage)
    {
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

        cv::Mat segmentMask;
        cv::inRange(hsv, cv::Scalar(lh, ls, lv), cv::Scalar(uh, us, uv), segmentMask);

        cv::Mat segmentedImage;

        image.copyTo(segmentedImage, segmentMask);

        cv::Mat grayScaleImage;
        cv::cvtColor(segmentedImage, grayScaleImage, cv::COLOR_BGR2GRAY);

        grayScaleImage.convertTo(storageImage, CV_32F);

        cv::imshow("segmented", storageImage);
        cv::imshow("gray", grayScaleImage);
    };
};

class DiscreteConvolution
{
private:
    cv::Mat kernel;

public:
    DiscreteConvolution(const cv::Mat &kernel) : kernel(kernel) {}

    cv::Mat conv(const cv::Mat &image)
    {
        cv::Mat output = cv::Mat::zeros(image.size(), CV_32F);

        int kernelRadiusX = kernel.cols / 2;
        int kernelRadiusY = kernel.rows / 2;
        // iterate over each pixel in the image, avoiding the edges to prevent accessing outside the image bounds.
        for (int y = kernelRadiusY; y < image.rows - kernelRadiusY; y++)
        {
            for (int x = kernelRadiusX; x < image.cols - kernelRadiusX; x++)
            {
                float sum = 0.0;
                // Iterate over the kernel.
                for (int k = -kernelRadiusY; k <= kernelRadiusY; k++)
                {
                    for (int l = -kernelRadiusX; l <= kernelRadiusX; l++)
                    {
                        // Multiply the kernel value by the corresponding image pixel value and add to sum.
                        // The kernel is centered on the current pixel (x, y).
                        sum += kernel.at<float>(k + kernelRadiusY, l + kernelRadiusX) *
                               image.at<float>(y + k, x + l);
                    }
                }
                output.at<float>(y, x) = sum;
            }
        }
        return output;
    }
};

class SobelDetector
{
private:
    DiscreteConvolution sobelX;
    DiscreteConvolution sobelY;

public:
    // constructor which accepts sobel kernels for X and Y directions
    SobelDetector(const cv::Mat &sobelXKernel, const cv::Mat &sobelYKernel)
        : sobelX(sobelXKernel), sobelY(sobelYKernel) {}

    // method to calculate Sobel edges
    void getEdges(const cv::Mat &grayScaleSegmentedImage, cv::Mat &sobledImage)
    {
        // apply sobel operator in X and Y directions
        cv::Mat sobelXResult = sobelX.conv(grayScaleSegmentedImage);
        cv::Mat sobelYResult = sobelY.conv(grayScaleSegmentedImage);

        // calculate gradient magnitude
        cv::Mat gradientMagnitude;
        cv::magnitude(sobelXResult, sobelYResult, gradientMagnitude);

        // normalize the gradient for display
        cv::normalize(gradientMagnitude, sobledImage, 0, 255, cv::NORM_MINMAX, CV_32FC1);

        cv::imshow("sobelX", sobelXResult);
        cv::imshow("sobelY", sobelYResult);
        cv::imshow("sobeled", sobledImage);
    }
};

class Hough
{
public:
    // Method to perform Otsu's thresholding
    void otsuThreshold(const cv::Mat &inputImage, cv::Mat &outputImage)
    {
        // Convert to 8-bit image for Otsu's thresholding if necessary
        cv::Mat input8U;
        if (inputImage.type() != CV_8U)
        {
            inputImage.convertTo(input8U, CV_8U);
        }
        else
        {
            input8U = inputImage;
        }

        // Apply Otsu's thresholding
        cv::threshold(input8U, outputImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        cv::imshow("Otsu Threshold", outputImage);
    }

    // Method to detect and draw lines using Hough Transform
    void detectAndDrawLines(const cv::Mat &inputImage, cv::Mat &srcImage)
    {
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(inputImage, lines, 1, CV_PI / 180, 50, 50, 10);

        // Draw the lines on the output image
        cv::Mat srcLinesImg = srcImage.clone();
        for (size_t i = 0; i < lines.size(); i++)
        {
            cv::Vec4i l = lines[i];
            cv::line(srcLinesImg, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        }

        cv::imshow("Detected Lines", srcLinesImg);
    }
};

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: ./main <path_to_image>\n";
        return -1;
    }

    CvBasic Processor;
    Storage Storage;
    DiscreteConvolution Convolver(Storage.gaussianKernel);
    SobelDetector SobelDetector(Storage.sobelXKernel, Storage.sobelYKernel);
    Hough HoughProcessor;
    // load image
    Storage.image = Processor.loadImage(argv[1]);
    // to adjust while running
    while (true)
    {
        // segment image
        Processor.segmentImage(Storage.image, Storage.graySegmentedImage);
        // convolute image
        Storage.convoloutedImage = Convolver.conv(Storage.graySegmentedImage);
        // sobel x and y
        SobelDetector.getEdges(Storage.convoloutedImage, Storage.sobledImage);

        HoughProcessor.otsuThreshold(Storage.sobledImage, Storage.otsuImage);

        HoughProcessor.detectAndDrawLines(Storage.otsuImage, Storage.image);

        char key = (char)cv::waitKey(30);
        if (key == 27)
            break;
    };
    return 0;
};