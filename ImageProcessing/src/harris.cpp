#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

void applyKernel(const Mat* const, Mat*, Mat);
void gaussianFilter(const Mat* const, Mat*);
void laplacianFilter(const Mat* const, Mat*, Mat*);
void sobelFilter(const Mat* const, Mat*, Mat*);
void harris(int, void*);
void eigen(const Mat * const, Mat*);

Mat src, srcGray;
int thrd = 1;
int maxThresh = 10;
char* sourceWindow = "Source image";
char* cornersWindow = "Corners detected";

int main(int argc, char** argv) {
	src = imread("C:\\Users\\Leandro\\Documents\\house.jpg", 1);
	cvtColor(src, srcGray, COLOR_BGR2GRAY);
	srcGray.convertTo(srcGray, CV_32FC(srcGray.channels()));

	namedWindow(sourceWindow, CV_WINDOW_AUTOSIZE);
	createTrackbar("Threshold: ", sourceWindow, &thrd, maxThresh, harris);
	convertScaleAbs(src, src);
	imshow(sourceWindow, src);
	
	harris(0, 0);
	waitKey(0);
	return 0;
}


void applyKernel(const Mat* const src, Mat* dst, Mat kernel)
{
	float sum[] = { 0.0f, 0.0f, 0.0f };
	int channels = src->channels();
	
	int nRows = src->rows;
	int nCols = src->cols;

	int offsetY = floor(kernel.rows*.5f);
	int offsetX = floor(kernel.cols*.5f);
	int currentI = 0, currentJ = 0;

	for (int i = 0; i < nRows; i++)
		for (int j = 0; j < nCols; j++) {
			

			
				for (int y = 0; y < kernel.rows; y++)
					for (int x = 0; x < kernel.cols; x++) {
						currentI = i - offsetY + y;
						currentJ = j - offsetX + x;

						if (currentI >= 0 && currentI < nRows && currentJ >= 0 && currentJ < nCols) {
							for (int channel = 0; channel < channels; channel++) {
								sum[channel] += kernel.at<float>(y, x) * src->at<float>(currentI, currentJ * channels + channel);
							}
						}

					}
				for (int channel = 0; channel < channels; channel++) {
					dst->at<float>(i, j*channels+channel) = sum[channel];
					sum[channel] = 0.0f;
				}
			
		}
	
}

void gaussianFilter(const Mat* const src, Mat* dst) {
	float norm = 1.0 / 64.0;
	Mat gaussian = (Mat_<float>(1, 7) << 1 * norm, 6 * norm, 15 * norm, 20 * norm, 15 * norm, 6 * norm, 1 * norm);

	Mat gaussianX = Mat::zeros(src->rows, src->cols, src->type());

	applyKernel(src, &gaussianX, gaussian);
	applyKernel(&gaussianX, dst, gaussian.t());
}
void laplacianFilter(const Mat* const src, Mat* dstX, Mat* dstY) {
	Mat lapX = (Mat_<float>(3, 3) << 0.0f, 1.0f, 0.0f, 1.0f, -4.0f, 1.0f, 0.0f, 1.0f, 0.0f);
	Mat lapY = (Mat_<float>(3, 3) << 1.0f, 1.0f, 1.0f, 1.0f, -8.0f, 1.0f, 1.0f, 1.0f, 1.0f);
	applyKernel(src, dstX, lapX);
	applyKernel(src, dstY, lapY);

}

void sobelFilter(const Mat* const src, Mat* dstX, Mat* dstY) {
	Mat sobelX = (Mat_<float>(3, 3) << 1.0f, 0.0f, -1.0f, 2.0f, 0.0f, -2.0f, 1.0f, 0.0f, -1.0f);
	Mat sobelY = (Mat_<float>(3, 3) << 1.0f, 2.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -2.0f, -1.0f);
	applyKernel(src, dstX, sobelX);
	applyKernel(src, dstY, sobelY);

}

void harris(int, void*) {
	int type = srcGray.type();
	int cols = srcGray.cols;
	int rows = srcGray.rows;

	Mat gaussian = Mat::zeros(rows, cols, type);
	gaussianFilter(&srcGray, &gaussian);

	Mat iX = Mat::zeros(rows, cols, type);
	Mat iY = Mat::zeros(rows, cols, type);
	Mat iXY = Mat::zeros(rows, cols, type);

	sobelFilter(&gaussian, &iX, &iY);
	multiply(iX, iY, iXY);
	multiply(iX, iX, iX);
	multiply(iY, iY, iY);

	Mat gaussianLoGX = Mat::zeros(rows, cols, type);
	Mat gaussianLoGY = Mat::zeros(rows, cols, type);
	Mat gaussianLoGXY = Mat::zeros(rows, cols, type);

	gaussianFilter(&iX, &gaussianLoGX);
	gaussianFilter(&iY, &gaussianLoGY);
	gaussianFilter(&iXY, &gaussianLoGXY);

	/*normalize(gaussianLoGX, gaussianLoGX, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	normalize(gaussianLoGY, gaussianLoGY, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	normalize(gaussianLoGXY, gaussianLoGXY, 0, 255, NORM_MINMAX, CV_32FC1, Mat());*/

	/*convertScaleAbs(gaussianLoGX, gaussianLoGX);
	convertScaleAbs(gaussianLoGY, gaussianLoGY);
	convertScaleAbs(gaussianLoGXY, gaussianLoGXY);*/


	Mat m = Mat::zeros(2, 2, type);
	Mat eigenValues = Mat::zeros(1, 2, CV_32F);
	Mat dstCircles = src.clone();
	for (int j = 0; j < rows; j++)
	{
		for (int i = 0; i < cols; i++)
		{

			m = (Mat_<float>(2, 2) << gaussianLoGX.at<float>(j, i), gaussianLoGXY.at<float>(j, i), gaussianLoGXY.at<float>(j, i), gaussianLoGY.at<float>(j, i));
			eigen(&m, &eigenValues); 

			if (eigenValues.at<float>(0) > thrd*10000 && eigenValues.at<float>(1) > thrd*10000)
			{
				circle(dstCircles, Point(i, j), 5, Scalar(0.0, 0.0, 255.0, 255.0), 2, 8, 0);
			}
		}
	}

	convertScaleAbs(dstCircles, dstCircles);
	imshow(sourceWindow, dstCircles);
}

void eigen(const Mat * const src, Mat* dst) {
	float b = -src->at<float>(0, 0) - src->at<float>(1, 1);
	float c = (-src->at<float>(0, 0) * -src->at<float>(1, 1)) - (-src->at<float>(0, 1) * -src->at<float>(1, 0));
	float squareRoot = sqrt(pow(b, 2) - 4 * c);
	dst->at<float>(0) = (-b + squareRoot)*0.5;
	dst->at<float>(1) = (-b - squareRoot)*0.5;
}
