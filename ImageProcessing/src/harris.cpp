#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <math.h>
#include <map>
#include <algorithm>  

using namespace cv;
using namespace std;

void applyKernel(const Mat* const, Mat*, Mat);
void gaussianFilter(const Mat* const, Mat*);
void laplacianFilter(const Mat* const, Mat*, Mat*);
void sobelFilter(const Mat* const, Mat*, Mat*);
void harris(int, void*);
void klt();
void eigen(const Mat * const, Mat*);
void onMouseCallback(int event, int x, int y, int flags, void* userData);

Mat src, srcGray;
int THRESHOLD = 96*10000;
char* sourceWindow = "Source image";
char* cornersWindow = "Corners detected";
int startPointX = -1;
int startPointY = -1;
Rect roi(0, 0, 1, 1);

struct MyPoint {
	MyPoint(int _x, int _y, int _cols) {
		x = _x;
		y = _y;
		cols = _cols;
	}

	MyPoint() {}

	int x;
	int y;
	int cols;
	bool operator() (const MyPoint& p1, const MyPoint& p2) const { return p1.x + p1.y*p1.cols < p2.x + p2.y*p2.cols; }
};

int main(int argc, char** argv) {
	/*MyPoint a(0,1,5);
	MyPoint b(0,0,5);
	MyPoint c(1,0,5);
	
	map<MyPoint, int, classcomp> points;

	points[a] = 1;
	points[b] = 2;
	points[c] = 3;

	cout << points[c] << endl;*/
	

	namedWindow(sourceWindow, CV_WINDOW_AUTOSIZE);
	namedWindow(cornersWindow, CV_WINDOW_AUTOSIZE);
	setMouseCallback(sourceWindow, onMouseCallback, 0);
	
	//convertScaleAbs(src, src);
	//imshow(sourceWindow, src);

	//VideoCapture cap(0); // open the default camera

	//if (!cap.isOpened())  // check if we succeeded
	//	return -1;

	//Mat edges;
	//namedWindow("edges", 1);

	//for (;;)
	//{
	//	Mat frame;
	//	cap >> frame; // get a new frame from camera
	//	cvtColor(frame, edges, CV_BGR2GRAY);
	//	edges.convertTo(edges, CV_32FC(edges.channels()));
	//	//GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
	//	//Canny(edges, edges, 0, 30, 3);
	//	gaussianFilter(&edges, &edges);
	//	convertScaleAbs(edges, edges);
	//	imshow("edges", edges);
	//	if (waitKey(30) >= 0) break;
	//}
	
	//harris(0,0);
	klt();
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

void sobelFilter(const Mat* const src, Mat* dstX, Mat* dstY) {
	Mat sobelX = (Mat_<float>(3, 3) << 1.0f, 0.0f, -1.0f, 2.0f, 0.0f, -2.0f, 1.0f, 0.0f, -1.0f);
	Mat sobelY = (Mat_<float>(3, 3) << 1.0f, 2.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -2.0f, -1.0f);
	applyKernel(src, dstX, sobelX);
	applyKernel(src, dstY, sobelY);

}

void harris(int , void*) {
	int type = srcGray.type();
	int cols = srcGray.cols;
	int rows = srcGray.rows;

	Mat iX = Mat::zeros(rows, cols, type);
	Mat iY = Mat::zeros(rows, cols, type);
	Mat iXY = Mat::zeros(rows, cols, type);

	/*Mat gaussian = Mat::zeros(rows, cols, type);
	gaussianFilter(&srcGray, &gaussian);*/

	sobelFilter(&srcGray, &iX, &iY);
	multiply(iX, iY, iXY);
	multiply(iX, iX, iX);
	multiply(iY, iY, iY);

	Mat gaussianX2 = Mat::zeros(rows, cols, type);
	Mat gaussianY2 = Mat::zeros(rows, cols, type);
	Mat gaussianXY = Mat::zeros(rows, cols, type);

	gaussianFilter(&iX, &gaussianX2);
	gaussianFilter(&iY, &gaussianY2);
	gaussianFilter(&iXY, &gaussianXY);

	/*normalize(gaussianLoGX, gaussianLoGX, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	normalize(gaussianLoGY, gaussianLoGY, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	normalize(gaussianLoGXY, gaussianLoGXY, 0, 255, NORM_MINMAX, CV_32FC1, Mat());*/

	/*convertScaleAbs(gaussianLoGX, gaussianLoGX);
	convertScaleAbs(gaussianLoGY, gaussianLoGY);
	convertScaleAbs(gaussianLoGXY, gaussianLoGXY);*/


	Mat m = Mat::zeros(2, 2, type);
	Mat eigenValues = Mat::zeros(1, 2, CV_32F);
	Mat dstCircles = src.clone();
	bool found = false;

	for (int j = roi.y; j < roi.height + roi.y; j++)
	{
		for (int i = roi.x; i < roi.width + roi.x; i++)
		{

			m = (Mat_<float>(2, 2) << gaussianX2.at<float>(j, i), gaussianXY.at<float>(j, i), gaussianXY.at<float>(j, i), gaussianY2.at<float>(j, i));
			eigen(&m, &eigenValues); 

			int R = (eigenValues.at<float>(0)*eigenValues.at<float>(1) - 0.06*pow(eigenValues.at<float>(0) + eigenValues.at<float>(1), 2))/10000;
			

			//if (eigenValues.at<float>(0) > thrd*10000 && eigenValues.at<float>(1) > thrd*10000)
			if(R > THRESHOLD)
			{
				circle(dstCircles, Point(i, j), 5, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);
			}
		}

		if (found) {
			j += 5;
			found = false;
		}


	}
	
	rectangle(dstCircles, roi, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);
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

void onMouseCallback(int event, int x, int y, int flags, void* userData) {
	if (event == EVENT_MOUSEMOVE && flags == EVENT_FLAG_LBUTTON) {
		if (startPointX < 0 && startPointY < 0) {
			startPointX = x;
			startPointY = y;
		} else {
			//Rect rect(startPointX, startPointY, x - startPointX, y - startPointY);
			roi.x = startPointX;
			roi.y = startPointY;
			roi.width = x - startPointX;
			roi.height = y - startPointY;

			/*Mat c = src.clone();

			rectangle(c, rect, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);

			imshow(sourceWindow, c);*/
			klt();
		}

	}

	if (event == EVENT_LBUTTONUP) {
		startPointX = startPointY = -1;
	}
	
	
}

void klt() {
	Mat srcA, srcAGray, srcB, srcBGray;

	srcA = imread("C:\\Users\\Leandro\\Documents\\a.png", 1);
	srcB = imread("C:\\Users\\Leandro\\Documents\\b.png", 1);

	cvtColor(srcA, srcAGray, COLOR_BGR2GRAY);
	cvtColor(srcB, srcBGray, COLOR_BGR2GRAY);

	srcAGray.convertTo(srcAGray, CV_32FC(srcAGray.channels()));
	srcBGray.convertTo(srcBGray, CV_32FC(srcBGray.channels()));

	int type = srcAGray.type();
	int cols = srcAGray.cols;
	int rows = srcAGray.rows;

	Mat iX = Mat::zeros(rows, cols, type);
	Mat iY = Mat::zeros(rows, cols, type);
	Mat iXY, t, tIx, tIy;

	//calcular gradiente em x e y	
	sobelFilter(&srcAGray, &iX, &iY);

	//calcular gradiente temporal t
	t = srcBGray - srcAGray;

	//executar as operações pixel-wise: Ix*Ix Iy*Iy Ix*Iy Ix*t Iy*t
	multiply(iX, iY, iXY);
	multiply(iX, iX, iX);
	multiply(iY, iY, iY);
	multiply(iX, t, tIx);
	multiply(iY, t, tIy);

	//passar um filtro gaussiano nos resultados das operações acima
	Mat gaussianX2 = Mat::zeros(rows, cols, type);
	Mat gaussianY2 = Mat::zeros(rows, cols, type);
	Mat gaussianXY = Mat::zeros(rows, cols, type);
	Mat gaussianTIx = Mat::zeros(rows, cols, type);
	Mat gaussianTIy = Mat::zeros(rows, cols, type);

	gaussianFilter(&iX, &gaussianX2);
	gaussianFilter(&iY, &gaussianY2);
	gaussianFilter(&iXY, &gaussianXY);
	gaussianFilter(&tIx, &gaussianTIx);
	gaussianFilter(&tIy, &gaussianTIy);

	//encontrar boas features
	Mat aTa = Mat::zeros(2, 2, type);
	Mat aTb = Mat::zeros(2, 2, type);
	Mat eigenValues = Mat::zeros(1, 2, CV_32F);
	map<MyPoint, float, MyPoint> points;
	Mat dstCircles = srcA.clone();

	for (int j = roi.y; j < roi.y + roi.height; j++)
	{
		for (int i = roi.x; i < roi.x+ roi.width; i++)
		{

			aTa = (Mat_<float>(2, 2) << gaussianX2.at<float>(j, i), gaussianXY.at<float>(j, i), gaussianXY.at<float>(j, i), gaussianY2.at<float>(j, i));
			
			

			if (determinant(aTa) != 0)
			{
				MyPoint p(i, j, cols);
				eigen(&aTa, &eigenValues);
				points[p] = min(eigenValues.at<float>(0), eigenValues.at<float>(1));
			}
		}
	}
	float maxLambda = max_element(points.begin(), points.end(), [](const pair<MyPoint, float>& p1, const pair<MyPoint, float>& p2) {
		return p1.second < p2.second; })->second;

	map<MyPoint, float, MyPoint> goodPoints;
	for (map<MyPoint, float>::iterator iterator = points.begin(); iterator != points.end(); iterator++) {
		float lambda = iterator->second;
		MyPoint p = iterator->first;
		if (lambda > maxLambda * 0.5) {
			goodPoints[p] = lambda;
		} 
	}

	map<MyPoint, float, MyPoint> filteredPoints;
	map<MyPoint, float, MyPoint> neighbours;
	for (map<MyPoint, float>::iterator iterator = goodPoints.begin(); iterator != goodPoints.end(); iterator++) {
		MyPoint current = iterator->first;
		
		
		for (int i = -1; i < 2; i++) {
			for (int j = -1; j < 2; j++) {
				map<MyPoint, float>::iterator it = goodPoints.find(MyPoint(current.x + j, current.y + i, cols));
				if (it != goodPoints.end()) {
					neighbours[it->first] = it->second;
				}
				
			}
		}


		map<MyPoint, float>::iterator maxLocal = max_element(neighbours.begin(), neighbours.end(), [](const pair<MyPoint, float>& p1, const pair < MyPoint, float>& p2) {
			return p1.second < p2.second;
		});
		filteredPoints[maxLocal->first] = maxLocal->second;
		neighbours.clear();
	}

	for (map<MyPoint, float>::iterator iterator = filteredPoints.begin(); iterator != filteredPoints.end(); iterator++) {
		MyPoint p = iterator->first;
		aTb = (Mat_<float>(2, 1) << gaussianTIx.at<float>(p.y, p.x), gaussianTIy.at<float>(p.y, p.x));
		aTa = (Mat_<float>(2, 2) << gaussianX2.at<float>(p.y, p.x), gaussianXY.at<float>(p.y, p.x), gaussianXY.at<float>(p.y, p.x), gaussianY2.at<float>(p.y, p.x));
		Mat v = aTa.inv()*aTb;
		//cout << v;
		circle(dstCircles, Point(p.x, p.y), 5, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);
		float radians = atan2(v.at<float>(1,0), v.at<float>(0,0));
		
		int x = cos(radians);
		int y = sin(radians);

		circle(srcB, Point(x + p.x, y + p.y), 5, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);
		
	}
	
	rectangle(dstCircles, roi, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);

	//desenhar o roi na nova imagem
	convertScaleAbs(dstCircles, dstCircles);
	convertScaleAbs(srcB, srcB);
	imshow(sourceWindow, dstCircles);
	imshow(cornersWindow, srcB);
}
