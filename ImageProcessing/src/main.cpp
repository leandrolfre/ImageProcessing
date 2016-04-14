#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include "Cimg.h"

using namespace cimg_library;

CImg<float> toGrayscale(const CImg<float> * const);
std::vector<CImg<float>> generatePyramid(const CImg<float> * const);
CImg<float> apply1DKernel(const CImg<float> * const, std::vector<float>);
CImg<float> downsample(const CImg<float> * const);
CImg<float> upsample(const CImg<float> * const);
CImg<float> reduce(const CImg<float> * const);
CImg<float> expand(const CImg<float> * const);
CImg<float> colapse(const CImg<float> * const, const CImg<float> * const);
//CImg<float> sum(const CImg<float> * const, const CImg<float> * const);
//CImg<float> minus(const CImg<float> * const, const CImg<float> * const);
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
		if(!maskImagePath.empty())
			imageMask.assign(maskImagePath.c_str());
		else {
			imageMask.assign(imageSource.width(), imageSource.height(), 1, 3, 0);
			for (int i = 0; i < imageSource.width()*0.5; i++)
				for (int j = 0; j < imageSource.height(); j++)
					imageMask(i, j, 0) = imageMask(i, j, 1) = imageMask(i, j, 2) = 255.0;
		}
		
		steps = fmin(steps, fmin(log2(imageSource.height()) - 1, log2(imageSource.width()) - 1));

		CImg<float> blended = blend(&imageSource, &imageBlend, &imageMask);
		std::string ext = cimg::split_filename(sourceImagePath.c_str());
		blended.save((outputPath + "\\blended." + ext).c_str());
	}
		
	
	
	/*std::cout << "Generating laplacian pyramid with " << steps << " levels." << std::endl;

	std::vector<CImg<float>> laplacianBuffer = generatePyramid(&imageSource);

	CImg<float> lastGaussianImg = laplacianBuffer.back();
	laplacianBuffer.pop_back();
	std::string ext = cimg::split_filename(sourceImagePath.c_str());
	for (int i = laplacianBuffer.size() - 1; i >= 0 ; i--) {
		CImg<float> collapsed = colapse(&lastGaussianImg, &laplacianBuffer[i]);
		if (debugMode) {
			collapsed.save((outputPath + "\\collapse." + ext).c_str(), i);
		}
		lastGaussianImg = collapsed;
	}*/
	
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

std::vector<CImg<float>> generatePyramid(const CImg<float> * const source) {

	std::vector<CImg<float>> laplacianBuffer;
	
	CImg<float> current = *source;
	
	for (int i = 0; i < steps; i++) {
		
		std::cout << "Reducing step " << i << std::endl;
		CImg<float> gaussianImg = reduce(&current);
		
		std::cout << "Expanding step " << i << std::endl;
		CImg<float> expanded = expand(&gaussianImg);
		std::cout << "Generating laplacian step " << i << std::endl;
		CImg<float> laplacianImg = current - expanded;//minus(&current, &expanded);

		current = gaussianImg;

		laplacianBuffer.push_back(laplacianImg);
		if (i == steps - 1) {
			laplacianBuffer.push_back(gaussianImg);
		}

		/*if (debugMode) {
			std::string ext = cimg::split_filename(sourceImagePath.c_str());
			source->save((outputPath + "\\original." + ext).c_str(), i);
			gaussianImg.save((outputPath + "\\gaussian." + ext).c_str(), i);
			expanded.save((outputPath + "\\expanded." + ext).c_str(), i);
			laplacianImg.save((outputPath + "\\laplacian." + ext).c_str(), i);
		}*/
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
	CImg<float> firstPass(width, height, 1, source->spectrum(), 255);
	CImg<float> secondPass(width, height, 1, source->spectrum(), 255);

	float magnitude[3] = { 0.0, 0.0, 0.0 };
	int offset = kernel.size() / 2;

	for (int j = 0; j < height; j++)
		for (int i = -offset; i < width - offset; i++) {

			for (int x = 0; x < kernel.size(); x++) {
				int offsetI = i+x;

				if (offsetI < 0) {
					offsetI = width + i + x;
				} else if (offsetI >= width) {
					offsetI = offsetI - width;
				}

				for (int k = 0; k < source->spectrum(); k++) {
					magnitude[k] += (kernel[x] * ((*source)(offsetI, j, k) / 255)) * 255;
				}
				
			}

			for (int k = 0; k < source->spectrum(); k++) {
				firstPass(i+offset, j, k) = magnitude[k];
			}
			magnitude[0] = 0.0;
			magnitude[1] = 0.0;
			magnitude[2] = 0.0;
		}
	//firstPass.save("C:\\Users\\Leandro\\Documents\\output_imgs\\firstPass.png", firstPass.width());
	for (int i = 0; i < width; i++)
		for (int j = -offset; j < height - offset; j++) {

			for (int x = 0; x < kernel.size(); x++) {
				int offsetJ = j + x;

				if (offsetJ < 0) {
					offsetJ = height + j + x;
				}
				else if (offsetJ >= height) {
					offsetJ = offsetJ - height;
				}

				for (int k = 0; k < firstPass.spectrum(); k++) {
					magnitude[k] += (kernel[x] * (firstPass(i, offsetJ, k) / 255)) * 255;
				}
			}

			for (int k = 0; k < firstPass.spectrum(); k++) {
				secondPass(i, j+offset, k) = magnitude[k];
			}
			magnitude[0] = 0.0;
			magnitude[1] = 0.0;
			magnitude[2] = 0.0;
		}
	//secondPass.save("C:\\Users\\Leandro\\Documents\\output_imgs\\secondPass.png", secondPass.width());
	return secondPass;
}

CImg<float> downsample(const CImg<float> * const source) {
	int width = source->width();
	int height = source->height();

	int halfWidth = width/2;
	int halfHeight = height/2;

	CImg<float> newImg(halfWidth, halfHeight, 1, 3, 255.0);

	for (int i = 0; i < halfWidth; i++) {

		for (int j = 0; j < halfHeight; j++) {

			for (int k = 0; k < source->spectrum(); k++) {
				newImg(i, j, 0, k) = (*source)(i*2, j*2, k);
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

	CImg<float> newImg(doubleWidth, doubleHeight, 1, 3, 0);

	for (int i = 0; i < width; i++) {

		for (int j = 0; j < height; j++) {

			//std::cout << "i:" << i << " j:" << j << std::endl;
			//td::cout << "x, y " << x << ", " << y << std::endl;

			for (int k = 0; k < source->spectrum(); k++) {
				newImg(i * 2, j * 2, k) = (*source)(i, j, k);
			}
			
		}
	}

	return newImg;

	//int width = source->width();
	//int height = source->height();

	//int doubleWidth = width * 2;
	//int doubleHeight = height * 2;

	//CImg<float> newImg(doubleWidth, doubleHeight, 1, 3, 255);

	//for (int i = 0; i < width; i++) {

	//	for (int j = 0; j < height; j++) {

	//		//std::cout << "i:" << i << " j:" << j << std::endl;
	//		//td::cout << "x, y " << x << ", " << y << std::endl;

	//		newImg(i * 2, j * 2, 0, 1) = (*source)(i, j, 1);
	//		newImg(i * 2, j * 2, 0, 2) = (*source)(i, j, 2);
	//		newImg(i * 2, j * 2, 0, 0) = (*source)(i, j, 0);

	//		newImg(i * 2 + 1, j * 2, 0, 1) = (*source)(i, j, 1);
	//		newImg(i * 2 + 1, j * 2, 0, 2) = (*source)(i, j, 2);
	//		newImg(i * 2 + 1, j * 2, 0, 0) = (*source)(i, j, 0);

	//		newImg(i * 2, j * 2 + 1, 0, 1) = (*source)(i, j, 1);
	//		newImg(i * 2, j * 2 + 1, 0, 2) = (*source)(i, j, 2);
	//		newImg(i * 2, j * 2 + 1, 0, 0) = (*source)(i, j, 0);

	//		newImg(i * 2 + 1, j * 2 + 1, 0, 1) = (*source)(i, j, 1);
	//		newImg(i * 2 + 1, j * 2 + 1, 0, 2) = (*source)(i, j, 2);
	//		newImg(i * 2 + 1, j * 2 + 1, 0, 0) = (*source)(i, j, 0);

	//	}
	//}

	//return newImg;
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

	//expd.save("C:\\Users\\Leandro\\Documents\\output_imgs\\expd.png",(*gaussian).width());
	CImg<float> collapsed = (*laplacian) + expd;// sum(laplacian, &expd);
	return collapsed;
}

//CImg<float> sum(const CImg<float> * const a, const CImg<float> * const b) {
//	int width = (*a).width();
//	int height = (*a).height();
//	CImg<float> result(width, height, 1, 3, 255.0);
//	for (int i = 0; i < width; i++) {
//		for (int j = 0; j < height; j++) {
//			result(i, j, 0) = (*a)(i, j, 0) + (*b)(i, j, 0);
//			float c = result(i, j, 0);
//			result(i, j, 1) = (*a)(i, j, 1) + (*b)(i, j, 1);
//			result(i, j, 2) = (*a)(i, j, 2) + (*b)(i, j, 2);
//		}
//	}
//
//	return result;
//}

//CImg<float> minus(const CImg<float> * const a, const CImg<float> * const b) {
//	int width = (*a).width();
//	int height = (*a).height();
//	CImg<float> result(width, height, 1, 3, 255.0);
//	for (int i = 0; i < width; i++) {
//		for (int j = 0; j < height; j++) {
//			result(i, j, 0) = (*a)(i, j, 0) - (*b)(i, j, 0);
//			result(i, j, 1) = (*a)(i, j, 1) - (*b)(i, j, 1);
//			result(i, j, 2) = (*a)(i, j, 2) - (*b)(i, j, 2);
//		}
//	}
//
//	return result;
//}

CImg<float> blend(const CImg<float> * const imgA, const CImg<float> * const imgB, const CImg<float> * const mask) {

	std::vector<CImg<float>> lbImgA = generatePyramid(imgA);
	std::vector<CImg<float>> lbImgB = generatePyramid(imgB);
	std::vector<CImg<float>> lbBlend;

	std::vector<CImg<float>> gbMask;

	CImg<float> currentMask = *mask;
	gbMask.push_back(currentMask);
	for (int i = 0; i < steps; i++) {
		gbMask.push_back(reduce(&currentMask));
		currentMask = gbMask[i + 1];
	}

	for (int i = 0; i < lbImgA.size(); i++) {
		std::cout << "Blending Step " << i << std::endl;
		lbBlend.push_back(CImg<float>(lbImgA[i].width(), lbImgA[i].height(), 1, 3, 255.0));

		for (int x = 0; x < lbImgA[i].width(); x++) {
			for (int y = 0; y < lbImgA[i].height(); y++) {

				/*for (int k = 0; k < 3; k++) {
					lbBlend[i].get_channel(k)(x,y) = ((gbMask[i].get_channel(k)(x, y) / 255.0)*(lbImgA[i].get_channel(k)(x, y) / 255.0) + (1 - (gbMask[i].get_channel(k)(x, y) / 255.0))*(lbImgB[i].get_channel(k)(x, y) / 255.0))*255.0;
					
				}*/

				float maskR = 0.0;
				float maskG = 0.0;
				float maskB = 0.0;

				if (gbMask[i].spectrum() == 1) {
					maskR = maskG = maskB = gbMask[i](x, y, 0);
				} else if (gbMask[i].spectrum() == 3) {
					maskR = gbMask[i](x, y, 0);
					maskG = gbMask[i](x, y, 1);
					maskB = gbMask[i](x, y, 2);
				}
				

				/*lbBlend[i](x, y, 0) = ((maskR * (lbImgA[i](x, y, 0) / 255.0)) + ((1 - maskR ) * (lbImgB[i](x, y, 0) / 255.0))) * 255.0;
				lbBlend[i](x, y, 1) = ((maskG * (lbImgA[i](x, y, 1) / 255.0)) + ((1 - maskG ) * (lbImgB[i](x, y, 1) / 255.0))) * 255.0;
				lbBlend[i](x, y, 2) = ((maskB * (lbImgA[i](x, y, 2) / 255.0)) + ((1 - maskB ) * (lbImgB[i](x, y, 2) / 255.0))) * 255.0;*/

				lbBlend[i](x, y, 0) = (maskR * lbImgA[i](x, y, 0) + (1 - maskR) * lbImgB[i](x, y, 0) );
				lbBlend[i](x, y, 1) = (maskG * lbImgA[i](x, y, 1) + (1 - maskG) * lbImgB[i](x, y, 1) );
				lbBlend[i](x, y, 2) = (maskB * lbImgA[i](x, y, 2) + (1 - maskB) * lbImgB[i](x, y, 2) );
			}
		}
	}
	if(debugMode){
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