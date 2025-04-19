#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
using namespace std;

int main(){
    cv::Mat img = cv::imread("C:/Users/DC Kong/Desktop/e24809bd330bce7429a74c16cc08f9d.png");
    cv::imshow("img",img);
    cv::waitKey(0);
    cout << "Hello World!"<<endl;
    return 0;
}