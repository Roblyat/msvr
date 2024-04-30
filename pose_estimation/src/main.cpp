//main.cpp
#include "../include/classCVBasic.h"
#include "../include/classStorage.h"
#include "../include/classSift.h"

//storage
Storage storage;

int main(){

    SIFT SIFT;
    SIFT.loadImage(storage.image);
    SIFT.siftExtract(storage.image, storage.img_with_keypoints, storage.keypoints);
    
    while (true) {
        //SIFT.trackbars();
        //SIFT.grayScale(storage.image, storage.gray_scaled_image);

        // Ergebnis anzeigen
        cv::namedWindow("SIFT Features", cv::WINDOW_NORMAL);
        //cv::imshow("sge.png", storage.image);
        cv::imshow("SIFT Features", storage.img_with_keypoints);

        SIFT.showKeyNumbers(storage.keypoints, storage.img_with_keypoints);

        cv::waitKey(0); // Warten, bis eine Taste gedrückt wird

        char key = (char) cv::waitKey(30);
        if (key == 27) break; 
    };

    return 0;

};