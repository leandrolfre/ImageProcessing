#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include "Cimg.h"

using namespace cimg_library;

CImg<float> toGrayscale(const CImg<float> * const);
void generatePyramid(const CImg<float> * const, std::string, std::string, int);
CImg<float> getBlur(const CImg<float> * const);
CImg<float> apply1DKernel(const CImg<float> * const, std::vector<float>);
CImg<float> downsample(const CImg<float> * const);
CImg<float> upsample(const CImg<float> * const);
CImg<float> reduce(const CImg<float> * const);
CImg<float> expand(const CImg<float> * const);
CImg<float> colapse(const CImg<float> * const);


int main(int argc, char **argv) {

	std::string sourceImagePath = cimg_option("-i", "", "Input image source");
	std::string outputPath = cimg_option("-o", ".\\", "Output image path");
	const int steps = cimg_option("-s", 1, "pyramid height");


	CImg<float> imageSource(sourceImagePath.c_str());
	std::string ext = cimg::split_filename(sourceImagePath.c_str());

	//std::cout << "source width:" << imageSource.width() << " height:" << imageSource.height() << std::endl;

	//CImg<float> grayscale = toGrayscale(&imageSource);
	//CImg<float> blurred = getBlur(&imageSource);

	//blurred.save((outputPath+"\\blurred."+ext).c_str());

	//generatePyramid(&imageSource, outputPyramidPath, ext, steps);

	CImg<float> reduced = reduce(&imageSource);
	CImg<float> expanded = expand(&reduced);
	CImg<float> minus = imageSource - expanded;
	CImg<float> sum = minus + expanded;

	reduced.save((outputPath + "\\reduced." + ext).c_str());
	expanded.save((outputPath + "\\expanded." + ext).c_str());
	minus.save((outputPath + "\\minus." + ext).c_str());
	sum.save((outputPath + "\\sum." + ext).c_str());






	return 0;
}

void generatePyramid(const CImg<float> * const source, std::string output, std::string ext, int steps) {
	int width = source->width();
	int height = source->height();

	int halfWidth = width*.5;
	int halfHeight = height*.5;

	if (steps < 1) return;

	try {
		CImg<float> newImg(halfWidth, halfHeight, 1, 3, 255);

		for (int i = 0; i < width; i += 2) {

			for (int j = 0; j < height; j += 2) {

				//std::cout << "i:" << i << " j:" << j << std::endl;
				//td::cout << "x, y " << x << ", " << y << std::endl;

				newImg(i*.5, j*.5, 0, 1) = (*source)(i, j, 1);
				newImg(i*.5, j*.5, 0, 2) = (*source)(i, j, 2);
				newImg(i*.5, j*.5, 0, 0) = (*source)(i, j, 0);
			}
		}


		newImg.save((output + "\\pyramid." + ext).c_str(), halfWidth);

		generatePyramid(&newImg, output, ext, steps - 1);

	}
	catch (CImgInstanceException e) {
		std::cout << e._message;
	}
}

CImg<float> toGrayscale(const CImg<float> * const source) {
	CImg<float> grayscaleLena(source->width(), source->height(), source->depth(), source->spectrum());
	for (int i = 0; i < source->width(); i++) {
		for (int j = 0; j < source->height(); j++) {
			const float
				valR = (*source)(i, j, 0),
				valG = (*source)(i, j, 1),
				valB = (*source)(i, j, 2),
				avg = (valR + valG + valB) / 3;

			grayscaleLena(i, j, 0) = grayscaleLena(i, j, 1) = grayscaleLena(i, j, 2) = avg;

		}
	}
	return grayscaleLena;
}

CImg<float> getBlur(const CImg<float> * const source) {
	int width = source->width();
	int height = source->height();
	CImg<float> blurredX(width, height, 1, 3, 255);
	CImg<float> blurredOne(width, height, 1, 3, 255);

	float fx = 1.0 / 16.0;
	float gaussianBlur[] = { 1 * fx, 4 * fx, 6 * fx, 4 * fx, 1 * fx };

	float magnitudeR = 0.0;
	float magnitudeG = 0.0;
	float magnitudeB = 0.0;
	int offset = 2;

	for (int j = 0; j < height; j++)
		for (int i = offset; i <= width - offset; i++) {

			for (int x = 0; x < 5; x++) {
				magnitudeR += gaussianBlur[x] * (*source)(i - offset + x, j, 0);
				magnitudeG += gaussianBlur[x] * (*source)(i - offset + x, j, 1);
				magnitudeB += gaussianBlur[x] * (*source)(i - offset + x, j, 2);
			}

			blurredX(i, j, 0) = magnitudeR;
			blurredX(i, j, 1) = magnitudeG;
			blurredX(i, j, 2) = magnitudeB;

			magnitudeR = 0.0;
			magnitudeG = 0.0;
			magnitudeB = 0.0;
		}

	for (int i = 0; i < width; i++)
		for (int j = offset; j <= height - offset; j++) {

			for (int x = 0; x < 5; x++) {
				magnitudeR += gaussianBlur[x] * blurredX(i, j - offset + x, 0);
				magnitudeG += gaussianBlur[x] * blurredX(i, j - offset + x, 1);
				magnitudeB += gaussianBlur[x] * blurredX(i, j - offset + x, 2);
			}

			blurredOne(i, j, 0) = magnitudeR;
			blurredOne(i, j, 1) = magnitudeG;
			blurredOne(i, j, 2) = magnitudeB;

			magnitudeR = 0.0;
			magnitudeG = 0.0;
			magnitudeB = 0.0;
		}

	return blurredOne;

}

CImg<float> apply1DKernel(const CImg<float> * const source, std::vector<float> kernel) {
	int width = source->width();
	int height = source->height();
	CImg<float> firstPass(width, height, 1, 3, 255);
	CImg<float> secondPass(width, height, 1, 3, 255);

	float magnitudeR = 0.0;
	float magnitudeG = 0.0;
	float magnitudeB = 0.0;
	int offset = 2;

	for (int j = 0; j < height; j++)
		for (int i = offset; i <= width - offset; i++) {

			for (int x = 0; x < kernel.size(); x++) {
				magnitudeR += (kernel[x] * ((*source)(i - offset + x, j, 0) / 255)) * 255;
				magnitudeG += (kernel[x] * ((*source)(i - offset + x, j, 1) / 255)) * 255;
				magnitudeB += (kernel[x] * ((*source)(i - offset + x, j, 2) / 255)) * 255;
			}

			firstPass(i, j, 0) = magnitudeR;
			firstPass(i, j, 1) = magnitudeG;
			firstPass(i, j, 2) = magnitudeB;

			magnitudeR = 0.0;
			magnitudeG = 0.0;
			magnitudeB = 0.0;
		}

	for (int i = 0; i < width; i++)
		for (int j = offset; j <= height - offset; j++) {

			for (int x = 0; x < kernel.size(); x++) {
				magnitudeR += (kernel[x] * (firstPass(i, j - offset + x, 0) / 255)) * 255;
				magnitudeG += (kernel[x] * (firstPass(i, j - offset + x, 1) / 255)) * 255;
				magnitudeB += (kernel[x] * (firstPass(i, j - offset + x, 2) / 255)) * 255;
			}

			secondPass(i, j, 0) = magnitudeR;
			secondPass(i, j, 1) = magnitudeG;
			secondPass(i, j, 2) = magnitudeB;

			magnitudeR = 0.0;
			magnitudeG = 0.0;
			magnitudeB = 0.0;
		}

	return secondPass;
}

CImg<float> downsample(const CImg<float> * const source) {
	int width = source->width();
	int height = source->height();

	int halfWidth = width*.5;
	int halfHeight = height*.5;

	CImg<float> newImg(halfWidth, halfHeight, 1, 3, 255);

	for (int i = 0; i < width; i += 2) {

		for (int j = 0; j < height; j += 2) {

			//std::cout << "i:" << i << " j:" << j << std::endl;
			//td::cout << "x, y " << x << ", " << y << std::endl;

			newImg(i*.5, j*.5, 0, 1) = (*source)(i, j, 1);
			newImg(i*.5, j*.5, 0, 2) = (*source)(i, j, 2);
			newImg(i*.5, j*.5, 0, 0) = (*source)(i, j, 0);
		}
	}

	return newImg;
}
CImg<float> upsample(const CImg<float> * const source) {
	int width = source->width();
	int height = source->height();

	int doubleWidth = width * 2;
	int doubleHeight = height * 2;

	CImg<float> newImg(doubleWidth, doubleHeight, 1, 3, 255);

	for (int i = 0; i < width; i++) {

		for (int j = 0; j < height; j++) {

			//std::cout << "i:" << i << " j:" << j << std::endl;
			//td::cout << "x, y " << x << ", " << y << std::endl;

			newImg(i * 2, j * 2, 0, 1) = (*source)(i, j, 1);
			newImg(i * 2, j * 2, 0, 2) = (*source)(i, j, 2);
			newImg(i * 2, j * 2, 0, 0) = (*source)(i, j, 0);

			newImg(i * 2 + 1, j * 2, 0, 1) = (*source)(i, j, 1);
			newImg(i * 2 + 1, j * 2, 0, 2) = (*source)(i, j, 2);
			newImg(i * 2 + 1, j * 2, 0, 0) = (*source)(i, j, 0);

			newImg(i * 2, j * 2 + 1, 0, 1) = (*source)(i, j, 1);
			newImg(i * 2, j * 2 + 1, 0, 2) = (*source)(i, j, 2);
			newImg(i * 2, j * 2 + 1, 0, 0) = (*source)(i, j, 0);

			newImg(i * 2 + 1, j * 2 + 1, 0, 1) = (*source)(i, j, 1);
			newImg(i * 2 + 1, j * 2 + 1, 0, 2) = (*source)(i, j, 2);
			newImg(i * 2 + 1, j * 2 + 1, 0, 0) = (*source)(i, j, 0);

		}
	}

	return newImg;
}

CImg<float> reduce(const CImg<float> * const source) {
	float weight = 1.0 / 16.0;
	float kernel[] = { 1 * weight, 4 * weight, 6 * weight, 4 * weight, 1 * weight };

	return downsample(&apply1DKernel(source, std::vector<float>(kernel, kernel + sizeof(kernel) / sizeof(kernel[0]))));

}
CImg<float> expand(const CImg<float> * const source) {
	float weight = 1.0 / 16.0;
	float kernel[] = { 1 * weight, 4 * weight, 6 * weight, 4 * weight, 1 * weight };

	return apply1DKernel(&upsample(source), std::vector<float>(kernel, kernel + sizeof(kernel) / sizeof(kernel[0])));
}
//CImg<float> colapse(const CImg<float> * const source) {}