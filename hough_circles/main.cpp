#include "iostream"
#include "opencv2/opencv.hpp"

struct Storage {   
    //images
    cv::Mat image;
    cv::Mat segmentedImage;
    cv::Mat convoloutedImage;
    cv::Mat graySegmentedImage;
    //vector for channels BGR
    std::vector<cv::Mat> channelsBGR;
    //mean kernal
    cv::Mat meanKernel = cv::Mat::ones(3,3, CV_32F)* 1/9;
    //gaussian kernal
    cv::Mat gaussianKernel = (cv::Mat_<float>(3,3) <<  1/16.0, 2/16.0, 1/16.0,
                                                       2/16.0, 4/16.0, 2/16.0,
                                                       1/16.0, 2/16.0, 1/16.0);
    //sobel X/Y
    cv::Mat sobelXKernel = (cv::Mat_<float>(3,3) << -1, 0, 1,
                                                    -2, 0, 2,
                                                    -1, 0, 1);

    cv::Mat sobelYKernel = (cv::Mat_<float>(3,3) <<
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1);
};

class CvBasic {
private:
    int lh = 0, uh = 8, ls = 90, us = 255, lv = 145, uv = 255;

public:
    //constructor
    CvBasic() {
    //initialize the segmentation window and sliders
    cv::namedWindow("Segmentation");
    cv::createTrackbar("Lower Hue", "Segmentation", &lh, 180);
    cv::createTrackbar("Upper Hue", "Segmentation", &uh, 180);
    cv::createTrackbar("Lower Saturation", "Segmentation", &ls, 255);
    cv::createTrackbar("Upper Saturation", "Segmentation", &us, 255);
    cv::createTrackbar("Lower Value", "Segmentation", &lv, 255);
    cv::createTrackbar("Upper Value", "Segmentation", &uv, 255);
    };
    //destructor
    ~CvBasic() {};

    //methods
    //load image & use clone to process
    cv::Mat loadImage(std::string imgPath) {
        cv::Mat image = cv::imread(imgPath).clone();
        return image;
    };
    //show image dimensions
    void showImgDimension(cv::Mat image) {
        std::cout << "[" << image.rows << "x" << image.cols << "x" << image.channels( ) << "]" << std::endl;
    };
    //convert the input BGR image to HSV color space for easier color segmentation
    cv::Mat segmentImage(const cv::Mat& image) {
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

        cv::Mat segmentMask;
        cv::inRange(hsv, cv::Scalar(lh, ls, lv), cv::Scalar(uh, us, uv), segmentMask);

        cv::Mat segmentedImage;

        image.copyTo(segmentedImage, segmentMask);

        cv::Mat grayScaleImage;
        cv::cvtColor(segmentedImage, grayScaleImage, cv::COLOR_BGR2GRAY);

        cv::Mat floatGrayScaledImage;
        grayScaleImage.convertTo(floatGrayScaledImage, CV_32F);

        cv::imshow("segmented", grayScaleImage);

    return grayScaleImage;
};

};

class DiscreteConvolution {
private:
    cv::Mat kernel;

public:
    DiscreteConvolution(const cv::Mat& kernel) : kernel(kernel) {}

    cv::Mat conv(const cv::Mat& image) {
        cv::Mat output = cv::Mat::zeros(image.size(), CV_32F);

        int kernelRadiusX = kernel.cols / 2;
        int kernelRadiusY = kernel.rows / 2;
        //iterate over each pixel in the image, avoiding the edges to prevent accessing outside the image bounds.
        for (int y = kernelRadiusY; y < image.rows - kernelRadiusY; y++) {
            for (int x = kernelRadiusX; x < image.cols - kernelRadiusX; x++) {
                float sum = 0.0;
                // Iterate over the kernel.
                for (int k = -kernelRadiusY; k <= kernelRadiusY; k++) {
                    for (int l = -kernelRadiusX; l <= kernelRadiusX; l++) {
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

class SobelDetector {
private:
    DiscreteConvolution sobelX;
    DiscreteConvolution sobelY;

public:
    //constructor which accepts sobel kernels for X and Y directions
    SobelDetector(const cv::Mat& sobelXKernel, const cv::Mat& sobelYKernel)
        : sobelX(sobelXKernel), sobelY(sobelYKernel) {}

    //method to calculate Sobel edges
    cv::Mat getEdges(const cv::Mat& grayScaleImage) {

        //apply sobel operator in X and Y directions
        cv::Mat sobelXResult = sobelX.conv(grayScaleImage);
        cv::Mat sobelYResult = sobelY.conv(grayScaleImage);

        //calculate gradient magnitude
        cv::Mat gradientMagnitude;
        cv::magnitude(sobelXResult, sobelYResult, gradientMagnitude);

        //normalize the gradient for display
        cv::Mat sobledImage;
        cv::normalize(gradientMagnitude, sobledImage, 0, 255, cv::NORM_MINMAX, CV_32FC1);
        
        cv::imshow("sobelX", sobelXResult);
        cv::imshow("sobelY", sobelYResult);
        cv::imshow("sobeled", sobledImage);

        return sobledImage;
    }
};

int main(int argc, char** argv) {

    if( argc != 2 ) {
        std::cout << "Usage: ./main <path_to_image>\n";
        return -1;                                       
    }

    CvBasic Processor;
    Storage Storage;
    DiscreteConvolution Convolver(Storage.gaussianKernel);
    SobelDetector SobelDetector(Storage.sobelXKernel, Storage.sobelYKernel);
    //load image
    Storage.image = Processor.loadImage(argv[1]);
    //to adjust while running
    while (true) {
        //segment image
        Storage.segmentedImage = Processor.segmentImage(Storage.image);
        //convolute image
        Storage.convoloutedImage = Convolver.conv(Storage.segmentedImage);        
        //sobel x and y 
        SobelDetector.getEdges(Storage.convoloutedImage);

        char key = (char) cv::waitKey(30);
        if (key == 27) break; 
    };

    return 0;

};