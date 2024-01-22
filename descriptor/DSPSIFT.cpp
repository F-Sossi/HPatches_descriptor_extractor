//
///**********************************************************************************************\
//Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/SIFT/
//Below is the original copyright.
//
////    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
////    All rights reserved.
//
//\**********************************************************************************************/

#include "DSPSIFT.h"
using namespace cv;
using namespace xfeatures2d;

//number of different scales to use. determined by factors in the range of [1/6, 4/3].
const int NUM_SCALES = 1; //CANNOT BE SET TO <= 0 
const double FLOAT_OCTAVE_SUB = 0;


// constructor
DSPSIFT::DSPSIFT()
{
	// do nothing
}

// This one acts just like SIFT...Should only be called by inherited classes that aren't using DSP - See below for DSP version.
void DSPSIFT::calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints,
	Mat& descriptors, int nOctaveLayers, int firstOctave) const
{
	int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

	int scalesWorkAround = NUM_SCALES; // used to avoid some divide by 0 compiler errors

	//used to hold calculated descriptors to average them later on
	Mat desc[NUM_SCALES];
	//used to hold the calculated factors 
	double yValue[NUM_SCALES];
	///testing liness
	double yv1 = 1;
	double yv2 = 1;

	/* Calculating points that fall on the line that is generated by yv1, and yv2 */
	if (NUM_SCALES == 1) //if the number of scales is 1, act like ordinary sift.
		yValue[0] = 1;
	else if(NUM_SCALES > 1)//else, resample the line between the points yv1, and yv2.
		for(int scale = 0; scale < NUM_SCALES; scale++)
			yValue[scale] = yv1 + scale * (yv2 - yv1) / (scalesWorkAround-1);

	for (int cs = 0; cs < NUM_SCALES; cs++) // 'current scale'
	{
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			KeyPoint kpt = keypoints[i];
			int octave, layer;
			float scale;

			
			///original calculations of octave, layer, scale, and size
			unpackOctave(kpt, octave, layer, scale);
			CV_Assert(octave >= firstOctave && layer <= nOctaveLayers + 2);
			float size = kpt.size*scale;
			size *= (float)yValue[cs];
			
			/*	This is used only for DSPSIFT
			//new calculations of octave, layer, scale, and size 
			float size = (float) (kpt.size * yValue[cs]);
			floatOctave = (float) (log2(size / (2 * sigma)) - FLOAT_OCTAVE_SUB);
			octave = (int) floor(floatOctave);
			layer = (int) floor((floatOctave - octave) * nOctaveLayers);
			scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
			size = size * scale;
			*/
			
			Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);

			//Gaussian Pyrmaid Info
			//ensures that the index accessed is a valid index of gpyr.
			int gpyrLength = (int) gpyr.size();
			int wantedIndex = ((octave - firstOctave)*(nOctaveLayers + 3) + layer);
			int correctedIndex = max(0, min(gpyrLength - 1, wantedIndex));
			/*
			If the octave we want to enter is less than -1 (negative index in gpyr),
			set a generic value as the descriptor for that key point, and go to the
			top of the loop.
			*/
			if (wantedIndex < 0)
			{
				//descriptors.at<float>((int)i, descC) = -1.f;
				for (int descC = 0; descC < descriptors.cols; descC++)
					descriptors.at<float>((int)i, descC) = -1.f;
				continue;
			}
			const Mat& img = gpyr[correctedIndex];

			float angle = 360.f - kpt.angle;
			if (std::abs(angle - 360.f) < FLT_EPSILON)
				angle = 0.f;

			calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
		}
		desc[cs] = descriptors.clone(); // keep a copy of the current scale
	}

	//average the descriptor values that resulted from the scales, and image pyramids.
	if (NUM_SCALES > 1)
	{
		int assignedScales = 0;
		for (int r = 0; r < descriptors.rows; r++)
		{
			for (int c = 0; c < descriptors.cols; c++)
			{
				assignedScales = 0;
				descriptors.at<float>(r, c) = 0;
				for (int cs = 0; cs < NUM_SCALES; cs++)
				{
					if (desc[cs].at<float>(r, c) != -1.f)
					{
						descriptors.at<float>(r, c) += desc[cs].at<float>(r, c);
						assignedScales++;
					}
				}
				descriptors.at<float>(r, c) /= assignedScales; 
			}
		}
	}
}


void DSPSIFT::calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints,
	Mat& descriptors, int nOctaveLayers, int firstOctave, int numScales, double linePoint1, double linePoint2) const
{
	int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

	std::vector<Mat> desc(numScales); //used to hold calculated descriptors to average them later on
	std::vector<double> yValue(numScales); //used to hold the calculated factors 	
	/* Calculating points that fall on the line that is generated by yv1, and yv2 */
	if (numScales == 1) //if the number of scales is 1, act like ordinary sift. [CURRENTLY TAKING MAX SCL FACTOR]
		yValue[0] = linePoint2;
	else if (numScales > 1)//else, resample the line between the points yv1, and yv2.
		for (int scale = 0; scale < numScales; scale++)
			yValue[scale] = linePoint1 + scale * (linePoint2 - linePoint1) / (numScales - 1);

	for (int cs = 0; cs < numScales; cs++) // 'current scale'
	{
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			KeyPoint kpt = keypoints[i];
			int octave, layer;
			float scale, floatOctave, size;

			if (numScales == 1) {
				///original calculations of octave, layer, scale, and size
				unpackOctave(kpt, octave, layer, scale);
				CV_Assert(octave >= firstOctave && layer <= nOctaveLayers + 2);
				size = kpt.size * scale;
				size *= (float)yValue[cs];
			}
			else {
				//calculations for octave, layer, scale, and size  
				size = (float)(kpt.size * yValue[cs]);
				floatOctave = (float)(log2(size / (2 * sigma)));// - FLOAT_OCTAVE_SUB);
				octave = (int)floor(floatOctave);
				layer = (int)floor((floatOctave - octave) * nOctaveLayers);
				scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
				size = size * scale;
			}

			Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);

			// Gaussian Pyramid Information
			//ensures that the index accessed is a valid index of gpyr.
			int gpyrLength = (int)gpyr.size();
			int wantedIndex = ((octave - firstOctave)*(nOctaveLayers + 3) + layer);
			int correctedIndex = max(0, min(gpyrLength - 1, wantedIndex));
	
			/*
			If the octave we want to enter is less than -1 (negative index in gpyr),
			set a generic value as the descriptor for that key point, and go to the
			/top of the loop.
			*/
			if (wantedIndex < 0)
			{
				//descriptors.at<float>((int)i) = -1.f;
				for (int descC = 0; descC < descriptors.cols; descC++)
					descriptors.at<float>((int)i, descC) = -1.f;
				continue;
			}
			const Mat& img = gpyr[correctedIndex];

			float angle = 360.f - kpt.angle;
			if (std::abs(angle - 360.f) < FLT_EPSILON)
				angle = 0.f;

			calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
		}
		desc[cs] = descriptors.clone(); // keep a copy of the current scale
	}

	//average the descriptor values that resulted from pooling across domains
	if (numScales > 1)
	{
		int assignedScales = 0;
		for (int r = 0; r < descriptors.rows; r++)
		{
			for (int c = 0; c < descriptors.cols; c++)
			{
				assignedScales = 0; 
				descriptors.at<float>(r, c) = 0;
				for (int cs = 0; cs < numScales; cs++)
				{
					if (desc[cs].at<float>(r, c) != -1.f) 
					{
						descriptors.at<float>(r, c) += desc[cs].at<float>(r, c); 
						assignedScales++;
					}
				}
				descriptors.at<float>(r, c) /= assignedScales;
			}
		}
	}
	yValue.clear();
	desc.clear();
}


//------------------------------------operator()---------------------------------------
// Overloading operator() to run the SIFT algorithm :
// 1. compute keypoints using local extrema of Dog space
// 2. compute descriptors with keypoints
//Precondition: the following parameters must be correctly defined.
//parameters:
//img: image base
//mask: image mask
//keypoints: keypoints of the image
//descriptors: descriptors
//useProvidedKeypoints: bool indicating whether using provided keypoints, false by default
//Postcondition: 1. keypoints are assigned
//				 2. descriptors are calculated and assigned
//-------------------------------------------------------------------------------------
void DSPSIFT::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& keypoints, OutputArray _descriptors,
	int numScales, double linePoint1, double linePoint2, bool useProvidedKeypoints) const
{
	int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
	Mat image = _image.getMat(), mask = _mask.getMat();

	if (image.empty() || image.depth() != CV_8U)
		CV_Error(CV_StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

	if (!mask.empty() && mask.type() != CV_8UC1)
		CV_Error(CV_StsBadArg, "mask has incorrect type (!=CV_8UC1)");

	if (useProvidedKeypoints)
	{
		firstOctave = 0; 
		int maxOctave = INT_MIN;
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			int octave, layer;
			float scale;
			unpackOctave(keypoints[i], octave, layer, scale);  //REPLACE WITH A FUNCTION THAT HAS OUR CALCULATIONS?
			firstOctave = std::min(firstOctave, octave); // ADDED: firstOctave is always <= 0
			maxOctave = std::max(maxOctave, octave); // ADDED: maxOctave is equal to the largest keypoint octave.
			actualNLayers = std::max(actualNLayers, layer - 2); // ADDED actualNlayers >= 0
		}

		firstOctave = std::min(firstOctave, 0);
		//CV_Assert(firstOctave >= -1 && actualNLayers <= nOctaveLayers);
		actualNOctaves = maxOctave - firstOctave + 1;
	}

	// base is a grey image
	Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);
	vector<Mat> gpyr, dogpyr;
	int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(log((double)std::min(base.cols, base.rows)) / log(2.) - 2) - firstOctave;

	// Expanding Gaussian Pyramid in case domain size pooling needs a higher octave
	if (numScales > 1 && linePoint1 < 1.0) nOctaves += 1;

	//double t, tf = getTickFrequency();
	//t = (double)getTickCount();
	buildGaussianPyramid(base, gpyr, nOctaves);
	buildDoGPyramid(gpyr, dogpyr);
	//t = (double)getTickCount() - t;
	//printf("pyramid construction time: %g\n", t*1000./tf);

	if (!useProvidedKeypoints)
	{
		//t = (double)getTickCount();
		findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
		KeyPointsFilter::removeDuplicated(keypoints);

		if (nfeatures > 0)
			KeyPointsFilter::retainBest(keypoints, nfeatures);
		//t = (double)getTickCount() - t;
		//printf("keypoint detection time: %g\n", t*1000./tf);

		if (firstOctave < 0)
			for (size_t i = 0; i < keypoints.size(); i++)
			{
				KeyPoint& kpt = keypoints[i];
				float scale = 1.f / (float)(1 << -firstOctave);
				kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
				kpt.pt *= scale;
				kpt.size *= scale;
			}

		if (!mask.empty())
			KeyPointsFilter::runByPixelsMask(keypoints, mask);
	}
	else
	{
		// filter keypoints by mask
		KeyPointsFilter::runByPixelsMask(keypoints, mask);
	}

	if (_descriptors.needed())
	{
		//t = (double)getTickCount();
		int dsize = descriptorSize();
		_descriptors.create((int)keypoints.size(), dsize, CV_32F);
		Mat descriptors = _descriptors.getMat();

		calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave,
			numScales, linePoint1, linePoint2);
		//t = (double)getTickCount() - t;
		//printf("descriptor extraction time: %g\n", t*1000./tf);
	}
}


//------------------------------------compute()----------------------------------------
// compute the descriptors with keypoints
//Precondition: the following parameters must be correctly defined.
//parameters:
//image: image base
//keypoints: calculated keypoints
//descriptors: descriptors to be computed
//Postcondition: descriptors are calculated and assigned
//-------------------------------------------------------------------------------------
void DSPSIFT::compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors, int numScales, double linePoint1, double linePoint2)
{
	(*this)(image, Mat(), keypoints, descriptors, numScales, linePoint1, linePoint2, true);
}


