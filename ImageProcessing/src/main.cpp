#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include "Cimg.h"

using namespace cimg_library;

std::vector<CImg<float>> generatePyramid(const CImg<float> * const);
CImg<float> apply1DKernel(const CImg<float> * const, std::vector<float>);
CImg<float> downsample(const CImg<float> * const);
CImg<float> upsample(const CImg<float> * const);
CImg<float> reduce(const CImg<float> * const);
CImg<float> expand(const CImg<float> * const);
CImg<float> colapse(const CImg<float> * const, const CImg<float> * const);
CImg<float> sum(const CImg<float> * const, const CImg<float> * const);
CImg<float> minus(const CImg<float> * const, const CImg<float> * const);
CImg<float> blend(const CImg<float> * const, const CImg<float> * const, const CImg<float> * const);

bool debugMode = false;
std::string sourceImagePath;
std::string blendImagePath;
std::string maskImagePath;
std::string outputPath;
int steps;

int main(int argc, char **argv) {

	sourceImagePath = cimg_option("-i", "", "Input image source");
	blendImagePath = cimg_option("-b", "", "Input blend image source");
	maskImagePath = cimg_option("-m", "", "Input mask image source");
	outputPath = cimg_option("-o", ".\\", "Output image path");
	steps = cimg_option("-s", 1, "pyramid height");
	debugMode = cimg_option("-debug", false, 0);

	if (!sourceImagePath.empty() && !blendImagePath.empty()) {
		CImg<float> imageSource(sourceImagePath.c_str());
		CImg<float> imageBlend(blendImagePath.c_str());
		CImg<float> imageMask;
		if (!maskImagePath.empty())
			imageMask.assign(maskImagePath.c_str());
		else {
			imageMask.assign(imageSource.width(), imageSource.height(), 1, 1, 0);
			for (int i = 0; i < imageSource.width()*0.5; i++)
				for (int j = 0; j < imageSource.height(); j++)
					imageMask(i, j, 0) = 255.0;
		}

		steps = fmin(steps, fmin(log2(imageSource.height()) - 1, log2(imageSource.width()) - 1));

		CImg<float> blended = blend(&imageSource, &imageBlend, &imageMask);
		std::string ext = cimg::split_filename(sourceImagePath.c_str());
		blended.save((outputPath + "\\blended." + ext).c_str());
	}

	return 0;
}

std::vector<CImg<float>> generatePyramid(const CImg<float> * const source) {

	std::vector<CImg<float>> laplacianBuffer;

	CImg<float> current = *source;
	CImg<float> gaussianImg;
	CImg<float> expanded;
	CImg<float> laplacianImg;
	for (int i = 0; i < steps; i++) {

		gaussianImg = reduce(&current);
		expanded = expand(&gaussianImg);
		laplacianImg = minus(&current, &expanded);
		current = gaussianImg;

		laplacianBuffer.push_back(laplacianImg);
		if (i == steps - 1) {
			laplacianBuffer.push_back(gaussianImg);
		}
	}

	return laplacianBuffer;
}

CImg<float> apply1DKernel(const CImg<float> * const source, std::vector<float> kernel) {
	int width = source->width();
	int height = source->height();
	CImg<float> firstPass(width, height, 1, source->spectrum(), 255.0);
	CImg<float> secondPass(width, height, 1, source->spectrum(), 255.0);

	float magnitude[3] = { 0.0, 0.0, 0.0 };
	int offset = kernel.size() / 2;

	for (int j = offset; j < height; j++) {
		for (int i = offset; i < width - offset - 1; i++) {

			for (int x = 0; x < kernel.size(); x++) {


				for (int k = 0; k < source->spectrum(); k++) {
					float a = (kernel[x]) * ((*source)(i - offset + x, j, k));
					magnitude[k] += a;
				}

			}

			for (int k = 0; k < source->spectrum(); k++) {
				firstPass(i, j, k) = magnitude[k];
			}
			magnitude[0] = 0.0;
			magnitude[1] = 0.0;
			magnitude[2] = 0.0;
		}
	}
	//firstPass.save((outputPath + "\\firstPass.jpg").c_str());

	for (int i = offset; i < width; i++) {
		for (int j = offset; j < height - offset - 1; j++) {

			for (int x = 0; x < kernel.size(); x++) {

				for (int k = 0; k < firstPass.spectrum(); k++) {
					magnitude[k] += ((kernel[x]) * (firstPass(i, j - offset + x, k)));
				}
			}

			for (int k = 0; k < firstPass.spectrum(); k++) {
				secondPass(i, j, k) = magnitude[k];
			}
			magnitude[0] = 0.0;
			magnitude[1] = 0.0;
			magnitude[2] = 0.0;
		}

	}
	//secondPass.save((outputPath + "\\secondPass.jpg").c_str());


	return secondPass;
}

CImg<float> downsample(const CImg<float> * const source) {
	int width = source->width();
	int height = source->height();

	int halfWidth = floor(width / 2.0);
	int halfHeight = floor(height / 2.0);

	CImg<float> newImg(halfWidth, halfHeight, 1, source->spectrum(), 0.0);

	for (int i = 0; i < halfWidth; i++) {

		for (int j = 0; j < halfHeight; j++) {

			for (int k = 0; k < source->spectrum(); k++) {
				newImg(i, j, 0, k) = (*source)(i * 2, j * 2, k);
			}

		}
	}

	return newImg;
}
CImg<float> upsample(const CImg<float> * const source) {
	int width = source->width();
	int height = source->height();

	int doubleWidth = width * 2;
	int doubleHeight = height * 2;

	CImg<float> newImg(doubleWidth, doubleHeight, 1, source->spectrum(), 0);

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < source->spectrum(); k++) {
				newImg(i * 2, j * 2, k) = (*source)(i, j, k);
			}

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
	float weight = 1.0 / 8.0;
	float kernel[] = { 1 * weight, 4 * weight, 6 * weight, 4 * weight, 1 * weight };
	return apply1DKernel(&upsample(source), std::vector<float>(kernel, kernel + sizeof(kernel) / sizeof(kernel[0])));
}

CImg<float> colapse(const CImg<float> * const gaussian, const CImg<float> * const laplacian) {
	CImg<float> expd = expand(gaussian);
	CImg<float> collapsed = sum(laplacian, &expd);
	return collapsed;
}

CImg<float> sum(const CImg<float> * const a, const CImg<float> * const b) {
	int width = a->width();
	int height = a->height();

	CImg<float> result(width, height, 1, b->spectrum(), 0.0);
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < b->spectrum(); k++) {
				result(i, j, k) = (*a)(i, j, k) + (*b)(i, j, k);
			}

		}
	}

	return result;
}

CImg<float> minus(const CImg<float> * const a, const CImg<float> * const b) {

	int width = a->width();
	int height = a->height();

	CImg<float> result(width, height, 1, b->spectrum(), 0.0);
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < b->spectrum(); k++) {
				result(i, j, k) = (*a)(i, j, k) - (*b)(i, j, k);
			}

		}
	}

	return result;
}

CImg<float> blend(const CImg<float> * const imgA, const CImg<float> * const imgB, const CImg<float> * const mask) {

	std::cout << "Generating Laplacian Pyramid A" << std::endl;
	std::vector<CImg<float>> lbImgA = generatePyramid(imgA);
	std::cout << "Generating Laplacian Pyramid B" << std::endl;
	std::vector<CImg<float>> lbImgB = generatePyramid(imgB);
	std::vector<CImg<float>> lbBlend;

	std::vector<CImg<float>> gbMask;
	std::cout << "Generating Gaussian Pyramid R" << std::endl;
	CImg<float> currentMask = *mask;
	gbMask.push_back(currentMask);
	for (int i = 0; i < steps; i++) {
		gbMask.push_back(reduce(&currentMask));
		currentMask = gbMask.back();
	}

	std::cout << "Generating Laplacian Pyramid L (Blending)" << std::endl;
	for (int i = 0; i < lbImgA.size(); i++) {
		lbBlend.push_back(CImg<float>(lbImgA[i].width(), lbImgA[i].height(), 1, 3, 0.0));
		gbMask[i] /= 255.0;

		for (int x = 0; x < lbImgA[i].width(); x++) {
			for (int y = 0; y < lbImgA[i].height(); y++) {

				float mask = gbMask[i](x, y, 0);

				for (int k = 0; k < lbImgA[i].spectrum(); k++) {
					lbBlend[i](x, y, k) = (mask * lbImgA[i](x, y, k) + ((1.0 - mask) * lbImgB[i](x, y, k)));
				}
			}
		}
	}

	if (debugMode) {
		CImg<float> lastGaussianImgA = lbImgA.back();
		lbImgA.pop_back();
		std::string extA = cimg::split_filename(sourceImagePath.c_str());
		for (int i = lbImgA.size() - 1; i >= 0; i--) {
			CImg<float> collapsed = colapse(&lastGaussianImgA, &lbImgA[i]);

			collapsed.save((outputPath + "\\collapseA." + extA).c_str(), i);
			lastGaussianImgA.save((outputPath + "\\lastGaussianImgA." + extA).c_str(), i);
			lbImgA[i].save((outputPath + "\\lbBlendA." + extA).c_str(), i);

			lastGaussianImgA = collapsed;
		}

		CImg<float> lastGaussianImgB = lbImgB.back();
		lbImgB.pop_back();
		std::string extB = cimg::split_filename(sourceImagePath.c_str());
		for (int i = lbImgB.size() - 1; i >= 0; i--) {
			CImg<float> collapsed = colapse(&lastGaussianImgB, &lbImgB[i]);

			collapsed.save((outputPath + "\\collapseB." + extB).c_str(), i);
			lastGaussianImgB.save((outputPath + "\\lastGaussianImgB." + extB).c_str(), i);
			lbImgB[i].save((outputPath + "\\lbBlendB." + extB).c_str(), i);

			lastGaussianImgB = collapsed;
		}

	}

	CImg<float> lastGaussianImg = lbBlend.back();
	lbBlend.pop_back();
	std::string ext = cimg::split_filename(sourceImagePath.c_str());
	std::cout << "Collapsing L" << std::endl;
	for (int i = lbBlend.size() - 1; i >= 0; i--) {
		CImg<float> collapsed = colapse(&lastGaussianImg, &lbBlend[i]);
		if (debugMode) {
			collapsed.save((outputPath + "\\collapse." + ext).c_str(), i);
			lastGaussianImg.save((outputPath + "\\lastGaussianImg." + ext).c_str(), i);
			lbBlend[i].save((outputPath + "\\lbBlend." + ext).c_str(), i);
		}
		lastGaussianImg = collapsed;
	}

	return lastGaussianImg;

}