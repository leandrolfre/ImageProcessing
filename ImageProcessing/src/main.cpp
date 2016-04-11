#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include "Cimg.h"

using namespace cimg_library;

CImg<float> toGrayscale(const CImg<float> * const);
std::vector<CImg<float>> generatePyramid(const CImg<float> * const, std::string, std::string, int);
CImg<float> apply1DKernel(const CImg<float> * const, std::vector<float>);
CImg<float> downsample(const CImg<float> * const);
CImg<float> upsample(const CImg<float> * const);
CImg<float> reduce(const CImg<float> * const);
CImg<float> expand(const CImg<float> * const);
CImg<float> colapse(const CImg<float> * const, const CImg<float> * const);
CImg<float> sum(const CImg<float> * const, const CImg<float> * const);
CImg<float> minus(const CImg<float> * const, const CImg<float> * const);


bool debugMode = false;


int main(int argc, char **argv) {

	std::string sourceImagePath = cimg_option("-i", "", "Input image source");
	std::string blendImagePath = cimg_option("-b", "", "Input blend image source");
	std::string maskImagePath = cimg_option("-m", "", "Input mask image source");
	std::string outputPath = cimg_option("-o", ".\\", "Output image path");
	int steps = cimg_option("-s", 1, "pyramid height");
	const bool debug = cimg_option("-debug", false, 0);
	
	CImg<float> imageSource(sourceImagePath.c_str());
	std::string ext = cimg::split_filename(sourceImagePath.c_str());

	debugMode = debug;
	steps = fmin(steps, log2(imageSource.width())-1);
	
	std::cout << "Generating laplacian pyramid with " << steps << " levels." << std::endl;

	std::vector<CImg<float>> laplacianBuffer = generatePyramid(&imageSource, outputPath, ext, steps);

	CImg<float> lastGaussianImg = laplacianBuffer.back();
	laplacianBuffer.pop_back();

	for (int i = laplacianBuffer.size() - 1; i >= 0 ; i--) {
		CImg<float> collapsed = colapse(&lastGaussianImg, &laplacianBuffer[i]);
		if (debugMode) {
			//lastGaussianImg.save((outputPath + "\\lastGaussianImg." + ext).c_str(), i);
			collapsed.save((outputPath + "\\collapse." + ext).c_str(), i);
		}
		lastGaussianImg = collapsed;
	}
	
	/*CImg<float> current = imageSource;

	for (int i = 0; i < 3; i++) {
		
		CImg<float> reduced = reduce(&current);
		CImg<float> expanded = expand(&reduced);
		CImg<float> laplacian = minus(&current,&expanded);
		CImg<float> laplacianR = sum(&laplacian, &expanded);
		

		reduced.save((outputPath + "\\reduced." + ext).c_str(), i);
		expanded.save((outputPath + "\\expanded." + ext).c_str(), i);
		laplacian.save((outputPath + "\\laplacian." + ext).c_str(), i);
		laplacianR.save((outputPath + "\\laplacianR." + ext).c_str(), i);
		current = reduced;


		
	}*/
	



	return 0;
}

std::vector<CImg<float>> generatePyramid(const CImg<float> * const source, std::string output, std::string ext, int steps) {

	std::vector<CImg<float>> laplacianBuffer;
	
	
	
	CImg<float> current = *source;
	
	for (int i = 0; i < steps; i++) {
		
		std::cout << "Generating gaussian step " << i << std::endl;
		CImg<float> gaussianImg = reduce(&current);
		
		CImg<float> expanded = expand(&gaussianImg);
		CImg<float> laplacianImg = minus(&current, &expanded);

		current = gaussianImg;

		laplacianBuffer.push_back(laplacianImg);
		if (i == steps - 1) {
			laplacianBuffer.push_back(gaussianImg);
		}

		if (debugMode) {
			gaussianImg.save((output + "\\gaussian." + ext).c_str(), i);
			expanded.save((output + "\\expanded." + ext).c_str(), i);
			laplacianImg.save((output + "\\laplacian." + ext).c_str(), i);
		}
	}
	
	return laplacianBuffer;
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

CImg<float> apply1DKernel(const CImg<float> * const source, std::vector<float> kernel) {
	int width = source->width();
	int height = source->height();
	CImg<float> firstPass(width, height, 1, 3, 255);
	CImg<float> secondPass(width, height, 1, 3, 255);

	float magnitudeR = 0.0;
	float magnitudeG = 0.0;
	float magnitudeB = 0.0;
	int offset = kernel.size()/2;

	for (int j = 0; j < height; j++)
		for (int i = offset; i < width - offset; i++) {

			for (int x = 0; x < kernel.size(); x++) {
				int offsetI = i - offset + x;

				magnitudeR += (kernel[x] * ((*source)(offsetI, j, 0) / 255)) * 255;
				magnitudeG += (kernel[x] * ((*source)(offsetI, j, 1) / 255)) * 255;
				magnitudeB += (kernel[x] * ((*source)(offsetI, j, 2) / 255)) * 255;
			}

			firstPass(i, j, 0) = magnitudeR;
			firstPass(i, j, 1) = magnitudeG;
			firstPass(i, j, 2) = magnitudeB;

			magnitudeR = 0.0;
			magnitudeG = 0.0;
			magnitudeB = 0.0;
		}

	for (int i = 0; i < width; i++)
		for (int j = offset; j < height - offset; j++) {
			
			for (int x = 0; x < kernel.size(); x++) {
				int offsetJ = j - offset + x;

				magnitudeR += (kernel[x] * (firstPass(i, offsetJ, 0) / 255)) * 255;
				magnitudeG += (kernel[x] * (firstPass(i, offsetJ, 1) / 255)) * 255;
				magnitudeB += (kernel[x] * (firstPass(i, offsetJ, 2) / 255)) * 255;
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

CImg<float> colapse(const CImg<float> * const gaussian, const CImg<float> * const laplacian) {
	CImg<float> expd = expand(gaussian);

	//expd.save("C:\\Users\\Leandro\\Documents\\output_imgs\\expd.gif",(*gaussian).width());
	CImg<float> collapsed = sum(laplacian, &expd);
	return collapsed;
}

CImg<float> sum(const CImg<float> * const a, const CImg<float> * const b) {
	int size = (*a).width();
	CImg<float> result(size, size, 1, 3, 255);
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			result(i, j, 0) = (*a)(i, j, 0) + (*b)(i, j, 0);
			result(i, j, 1) = (*a)(i, j, 1) + (*b)(i, j, 1);
			result(i, j, 2) = (*a)(i, j, 2) + (*b)(i, j, 2);
		}
	}

	return result;
}

CImg<float> minus(const CImg<float> * const a, const CImg<float> * const b) {
	int size = (*a).width();
	CImg<float> result(size, size, 1, 3, 255);
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			result(i, j, 0) = (*a)(i, j, 0) - (*b)(i, j, 0);
			result(i, j, 1) = (*a)(i, j, 1) - (*b)(i, j, 1);
			result(i, j, 2) = (*a)(i, j, 2) - (*b)(i, j, 2);
		}
	}

	return result;
}

