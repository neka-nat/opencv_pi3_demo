#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;

int main(int argc, const char * argv[])
{
    /* input image */
    Mat src = imread("../lena.jpg", IMREAD_UNCHANGED);
    resize(src, src, Size(), 2, 2);
    UMat usrc;
    src.copyTo(usrc);
    UMat udst, ufilt;
    cvtColor(usrc, udst, CV_BGR2GRAY);
    auto start = getTickCount();
    bilateralFilter(udst, ufilt, 8, 15, 15);
    auto end = getTickCount();
    double elapsedmsec = (end - start) * 1000 / getTickFrequency();
    std::cout << elapsedmsec << std::endl;
    imshow("lena_filt", ufilt);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
