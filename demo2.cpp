#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <vector>
#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, const char** argv) {
    VideoCapture vcap(0);
    CascadeClassifier fd("haarcascade_frontalface_default.xml");
    Mat frame, frameGray;
    //UMat frame, frameGray;
    vector<Rect> faces;
    for(;;) {
        // processing loop
        vcap >> frame;
        auto start = getTickCount();
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        equalizeHist(frameGray, frameGray);
        fd.detectMultiScale(frameGray, faces);
        auto end = getTickCount();
        double elapsedmsec = (end - start) * 1000 / getTickFrequency();
        std::cout << elapsedmsec << std::endl;
        for (size_t i = 0; i < faces.size(); i++) {
            Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
            ellipse(frame, center, Size(faces[i].width / 2 + 10, faces[i].height / 2 + 10), 0, 0, 360, Scalar(0, 0, 255), 3);
        }
        imshow("Face detection", frame);
        waitKey(1);
    }
}

