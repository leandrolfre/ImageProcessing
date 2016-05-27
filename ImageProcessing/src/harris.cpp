#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <math.h>
#include <map>
#include <algorithm>  

using namespace cv;
using namespace std;

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
	bool operator() (const MyPoint& p1, const MyPoint& p2) const { return p1.y + p1.x*p1.cols < p2.y + p2.x*p2.cols; }
};

void applyKernel(const Mat* const, Mat*, Mat);
void gaussianFilter(const Mat* const, Mat*);
void laplacianFilter(const Mat* const, Mat*, Mat*);
void sobelFilter(const Mat* const, Mat*, Mat*);
void harris(int, void*);
vector<MyPoint> getGoodPoints(const Mat* const x2, const Mat* const xy, const Mat* const y2);
vector<MyPoint> getFlow(const vector<MyPoint>* const goodPoints, const Mat* const x2, const Mat* const xy, const Mat* const y2, const Mat* const tx, const Mat* const ty);
vector<MyPoint> klt(const vector<MyPoint>* const, const Mat* const, const Mat* const, const Mat* const, const Mat* const, const Mat* const);
void eigen(const Mat * const, Mat*);
void onMouseCallback(int event, int x, int y, int flags, void* userData);
Mat downsample(const Mat * const);
Mat reduce(const Mat * const);
std::vector<Mat> generatePyramid(const Mat * const, int);

Mat srcA, srcAGray, srcB, srcBGray;
int THRESHOLD = 96*10000;
char* sourceWindow = "Source image";
char* cornersWindow = "Corners detected";
int startPointX = -1;
int startPointY = -1;
VideoCapture videoInput;
VideoWriter videoOutput;
Rect roi(0, 0, 1, 1);



int main(int argc, char** argv) {
	
	namedWindow(sourceWindow, CV_WINDOW_AUTOSIZE);
	//namedWindow(cornersWindow, CV_WINDOW_AUTOSIZE);
	setMouseCallback(sourceWindow, onMouseCallback, 0);
	videoInput.open("C:\\Users\\Leandro\\Documents\\car\\%04d.jpg");
	videoOutput.open("C:\\Users\\Leandro\\Documents\\car\\tracker.mp4", CV_FOURCC('F', 'M', 'P', '4'), 30.0, cvSize(videoInput.get(CAP_PROP_FRAME_WIDTH), videoInput.get(CAP_PROP_FRAME_HEIGHT)));

	videoInput >> srcA;
	cvtColor(srcA, srcAGray, COLOR_BGR2GRAY);
	srcAGray.convertTo(srcAGray, CV_32FC1);

	int type = srcAGray.type();
	int cols = srcAGray.cols;
	int rows = srcAGray.rows;

	Mat iX = Mat::zeros(rows, cols, type);
	Mat iY = Mat::zeros(rows, cols, type);
	Mat iX2 = Mat::zeros(rows, cols, type);
	Mat iY2 = Mat::zeros(rows, cols, type);
	Mat gaussianX2 = Mat::zeros(rows, cols, type);
	Mat gaussianY2 = Mat::zeros(rows, cols, type);
	Mat gaussianXY = Mat::zeros(rows, cols, type);
	Mat gaussianTIx = Mat::zeros(rows, cols, type);
	Mat gaussianTIy = Mat::zeros(rows, cols, type);
	Mat iXY, t, tIx, tIy;
	
	imshow(sourceWindow, srcA);
	while (waitKey(0) != 13) {}

	sobelFilter(&srcAGray, &iX, &iY);
	multiply(iX, iY, iXY);
	multiply(iX, iX, iX2);
	multiply(iY, iY, iY2);

	gaussianFilter(&iX2, &gaussianX2);
	gaussianFilter(&iY2, &gaussianY2);
	gaussianFilter(&iXY, &gaussianXY);
	
	vector<MyPoint> goodPoints = getGoodPoints(&gaussianX2, &gaussianXY, &gaussianY2);
	
	rectangle(srcA, roi, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);
	videoOutput << srcA;
	

	while (videoInput.read(srcB)) {
		
		cvtColor(srcB, srcBGray, COLOR_BGR2GRAY);
		srcBGray.convertTo(srcBGray, CV_32FC1);
		t = srcBGray - srcAGray;
		multiply(iX, t, tIx);
		multiply(iY, t, tIy);
		gaussianFilter(&tIx, &gaussianTIx);
		gaussianFilter(&tIy, &gaussianTIy);

		goodPoints = klt(&goodPoints, &gaussianX2, &gaussianXY, &gaussianY2, &gaussianTIx, &gaussianTIy);

		float x = 0.0, y = 0.0;
		Mat srcBClone = srcB.clone();
		for (int i = 0; i < goodPoints.size(); i++) {
			/*x += goodPoints[i].x;
			y += goodPoints[i].y;*/
			//desenha novo roi
			/*roi.x += x;
			roi.y += y;*/
			//rectangle(srcBClone, roi, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);
			circle(srcBClone, Point(goodPoints[i].y, goodPoints[i].x), 1, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);
		}

		/*x /= goodPoints.size();
		y /= goodPoints.size();*/

		
		imshow(sourceWindow, srcBClone);
		waitKey(30);
		videoOutput << srcBClone;

		//atualiza imagens
		srcAGray = srcBGray.clone();
		sobelFilter(&srcAGray, &iX, &iY);
		multiply(iX, iY, iXY);
		multiply(iX, iX, iX2);
		multiply(iY, iY, iY2);

		gaussianFilter(&iX2, &gaussianX2);
		gaussianFilter(&iY2, &gaussianY2);
		gaussianFilter(&iXY, &gaussianXY);

	}

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

//void harris(int , void*) {
//	int type = srcGray.type();
//	int cols = srcGray.cols;
//	int rows = srcGray.rows;
//
//	Mat iX = Mat::zeros(rows, cols, type);
//	Mat iY = Mat::zeros(rows, cols, type);
//	Mat iXY = Mat::zeros(rows, cols, type);
//
//	/*Mat gaussian = Mat::zeros(rows, cols, type);
//	gaussianFilter(&srcGray, &gaussian);*/
//
//	sobelFilter(&srcGray, &iX, &iY);
//	multiply(iX, iY, iXY);
//	multiply(iX, iX, iX);
//	multiply(iY, iY, iY);
//
//	Mat gaussianX2 = Mat::zeros(rows, cols, type);
//	Mat gaussianY2 = Mat::zeros(rows, cols, type);
//	Mat gaussianXY = Mat::zeros(rows, cols, type);
//
//	gaussianFilter(&iX, &gaussianX2);
//	gaussianFilter(&iY, &gaussianY2);
//	gaussianFilter(&iXY, &gaussianXY);
//
//	/*normalize(gaussianLoGX, gaussianLoGX, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
//	normalize(gaussianLoGY, gaussianLoGY, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
//	normalize(gaussianLoGXY, gaussianLoGXY, 0, 255, NORM_MINMAX, CV_32FC1, Mat());*/
//
//	/*convertScaleAbs(gaussianLoGX, gaussianLoGX);
//	convertScaleAbs(gaussianLoGY, gaussianLoGY);
//	convertScaleAbs(gaussianLoGXY, gaussianLoGXY);*/
//
//
//	Mat m = Mat::zeros(2, 2, type);
//	Mat eigenValues = Mat::zeros(1, 2, CV_32F);
//	Mat dstCircles = src.clone();
//	bool found = false;
//
//	for (int j = roi.y; j < roi.height + roi.y; j++)
//	{
//		for (int i = roi.x; i < roi.width + roi.x; i++)
//		{
//
//			m = (Mat_<float>(2, 2) << gaussianX2.at<float>(j, i), gaussianXY.at<float>(j, i), gaussianXY.at<float>(j, i), gaussianY2.at<float>(j, i));
//			eigen(&m, &eigenValues); 
//
//			int R = (eigenValues.at<float>(0)*eigenValues.at<float>(1) - 0.06*pow(eigenValues.at<float>(0) + eigenValues.at<float>(1), 2))/10000;
//			
//
//			//if (eigenValues.at<float>(0) > thrd*10000 && eigenValues.at<float>(1) > thrd*10000)
//			if(R > THRESHOLD)
//			{
//				circle(dstCircles, Point(i, j), 5, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);
//			}
//		}
//
//		if (found) {
//			j += 5;
//			found = false;
//		}
//
//
//	}
//	
//	rectangle(dstCircles, roi, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);
//	convertScaleAbs(dstCircles, dstCircles);
//
//	imshow(sourceWindow, dstCircles);
//}

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
			Rect rect(startPointX, startPointY, x - startPointX, y - startPointY);
			roi.x = startPointX;
			roi.y = startPointY;
			roi.width = x - startPointX;
			roi.height = y - startPointY;

			Mat c = srcA.clone();

			rectangle(c, rect, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);

			imshow(sourceWindow, c);
			
		}

	}

	if (event == EVENT_LBUTTONUP) {
		startPointX = startPointY = -1;
	}
	
	
}

vector<MyPoint> getGoodPoints(const Mat* const x2, const Mat* const xy, const Mat* const y2) {
	int cols, rows, type;
	cols = x2->cols;
	rows = x2->rows;
	type = x2->type();


	Mat aTa = Mat::zeros(2, 2, type);
	Mat eigenValues = Mat::zeros(1, 2, CV_32F);
	map<MyPoint, float, MyPoint> points;
	
	

	for (int j = roi.y; j < roi.y + roi.height; j++)
	{
		for (int i = roi.x; i < roi.x + roi.width; i++)
		{

			aTa = (Mat_<float>(2, 2) << x2->at<float>(j, i), xy->at<float>(j, i), xy->at<float>(j, i), y2->at<float>(j, i));
			eigen(&aTa, &eigenValues);


			if (eigenValues.at<float>(0) > 0 && eigenValues.at<float>(1) > 0)
			{
				MyPoint p(j, i, cols);
				points[p] = min(eigenValues.at<float>(0), eigenValues.at<float>(1));
			}
		}
	}
	map<MyPoint, float, MyPoint>::iterator maxLambdaIt = max_element(points.begin(), points.end(), [](const pair<MyPoint, float>& p1, const pair<MyPoint, float>& p2) {
		return p1.second <= p2.second; });

	float maxLambda = (maxLambdaIt != points.end() ? maxLambdaIt->second : 0.0f);
	map<MyPoint, float, MyPoint> goodPoints;
	for (map<MyPoint, float>::iterator iterator = points.begin(); iterator != points.end(); iterator++) {
		float lambda = iterator->second;
		MyPoint p = iterator->first;
		if (lambda > maxLambda * 0.1) {
			goodPoints[p] = lambda;
		}
	}

	vector<MyPoint> filteredPoints;
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
			return p1.second <= p2.second;
		});

		filteredPoints.push_back(maxLocal->first);


		neighbours.clear();
	}

	return filteredPoints;
}

vector<MyPoint> getFlow(const vector<MyPoint>* const goodPoints, const Mat* const x2, const Mat* const xy, const Mat* const y2, const Mat* const tx, const Mat* const ty) {

	int type = x2->type();

	Mat v = Mat::zeros(2, 1, type);
	Mat aTa = Mat::zeros(2, 2, type);
	Mat aTb = Mat::zeros(2, 1, type);
	float x = 0;
	float y = 0;

	int levels = 6;

	vector<Mat> vGaussianTIx = generatePyramid(tx, levels);
	vector<Mat> vGaussianTIy = generatePyramid(ty, levels);
	vector<Mat> vGaussianX2 = generatePyramid(x2, levels);
	vector<Mat> vGaussianXY = generatePyramid(xy, levels);
	vector<Mat> vGaussianY2 = generatePyramid(y2, levels);

	vector<MyPoint> flowPoints;

	//calcular o fluxo
	for (vector<MyPoint>::size_type point = 0; point < goodPoints->size(); point++) {
		MyPoint p = goodPoints->at(point);

		Mat lastFlow = Mat::zeros(2, 1, type);
		for (int l = levels; l >= 0; l--) {
			int level = pow(2, l);
			aTa = (Mat_<float>(2, 2) << vGaussianX2[l].at<float>(p.x / level, p.y / level), vGaussianXY[l].at<float>(p.x / level, p.y / level), vGaussianXY[l].at<float>(p.x / level, p.y / level), vGaussianY2[l].at<float>(p.x / level, p.y / level));
			aTb = (Mat_<float>(2, 1) << -vGaussianTIx[l].at<float>(p.x / level, p.y / level), -vGaussianTIy[l].at<float>(p.x / level, p.y / level));

			/*cout << "lastFlow - " << lastFlow << endl;
			cout << "2*lastFlow - " << (2 * lastFlow) << endl;
			cout << "flow - " << (aTa.inv()*(aTb)) << endl;*/
			Mat currentFlow = (aTa.inv()*(aTb)) + (2 * lastFlow);
			//cout << "currentFlow - " << currentFlow << endl;
			lastFlow = currentFlow;

		}
		//circle(dstCircles, Point(p.y, p.x), 1, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);
		double rad = atan2(lastFlow.at<float>(1, 0), lastFlow.at<float>(0, 0));
		x = cos(rad);
		y = sin(rad)*-1;
		/*Point newPoint(p.y + lastFlow.at<float>(0, 0)*-1.0, p.x + lastFlow.at<float>(1, 0)*-1.0);
		Point current(p.y, p.x);

		//debug
		cout << "lastFlow: " << lastFlow << endl;
		cout << "current: " << current << endl;
		cout << "newPoint: " << newPoint << endl;*/
		/*		arrowedLine(dstCircles, current, Point(p.y + x*15.0, p.x + y*15.0), Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);*/
		//circle(srcB, Point(p.y + x, p.x + y), 1, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);


		flowPoints.push_back(MyPoint(p.x + lastFlow.at<float>(1, 0)*-1.0, p.y + lastFlow.at<float>(0, 0),0));
		//cout << "lastFlow: " << lastFlow << endl;
		//flowPoints.push_back(MyPoint(p.x + y, p.y + x, 0));
	}

	return flowPoints;
}

vector<MyPoint> klt(const vector<MyPoint>* const points, const Mat* const x2, const Mat* const xy, const Mat* const y2, const Mat* const tx, const Mat* const ty) {
	
	vector<MyPoint> updatedPoints = getFlow(points, x2, xy, y2, tx, ty);
	

	////calcular o fluxo
	//for (map<MyPoint, float>::iterator iterator = filteredPoints.begin(); iterator != filteredPoints.end(); iterator++) {
	//	MyPoint p = iterator->first;
	//	aTb = (Mat_<float>(2, 1) << -gaussianTIx.at<float>(p.x, p.y), -gaussianTIy.at<float>(p.x, p.y));
	//	aTa = (Mat_<float>(2, 2) << gaussianX2.at<float>(p.x, p.y), gaussianXY.at<float>(p.x, p.y), gaussianXY.at<float>(p.x, p.y), gaussianY2.at<float>(p.x, p.y));
	//	Mat u = aTa.inv()*(aTb);
	//	
	//	circle(dstCircles, Point(p.y, p.x), 1, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);
	//	double rad = atan2(u.at<float>(1, 0), u.at<float>(0, 0));
	//	x = round(cos(rad));
	//	y = round(sin(rad)*-1);
	//	Point newPoint(p.y+x, p.x+y);
	//	Point current(p.y, p.x);

	//	/*cout << current << endl;
	//	cout << newPoint << endl;*/
	//	arrowedLine(dstCircles, current, Point(p.y + x*15.0, p.x + y*15.0), Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);
	//	circle(srcB, Point(p.y + x, p.x + y), 1, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);
	//}

	/*x /= filteredPoints.size();
	y /= filteredPoints.size();
	cout << " x:" << x << " y:" << y << endl;
	rectangle(dstCircles, roi, Scalar(0.0, 0.0, 255.0, 255.0), 1, CV_AA, 0);*/

	return updatedPoints;
	
}

Mat downsample(const Mat * const source) {
	int cols = source->cols;
	int rows = source->rows;
	int type = source->type();

	int halfCols = floor(cols / 2.0);
	int halfRows = floor(rows / 2.0);

	Mat newImg = Mat::zeros(halfRows, halfCols, type);

	for (int i = 0; i < halfCols; i++) {

		for (int j = 0; j < halfRows; j++) {
			newImg.at<float>(j, i) = (*source).at<float>(j * 2, i * 2);
		}
	}

	return newImg;
}

Mat reduce(const Mat * const source) {
	int cols = source->cols;
	int rows = source->rows;
	int type = source->type();

	Mat dst = Mat::zeros(rows, cols, type);

	gaussianFilter(source, &dst);
	return downsample(&dst);

}

std::vector<Mat> generatePyramid(const Mat * const source, int levels) {

	std::vector<Mat> gaussianBuffer;

	Mat current = source->clone();
	Mat gaussianImg;
	gaussianBuffer.push_back(current);
	for (int i = 0; i < levels; i++) {

		gaussianImg = reduce(&current);
		current = gaussianImg;

		gaussianBuffer.push_back(gaussianImg);

	}

	return gaussianBuffer;
}

