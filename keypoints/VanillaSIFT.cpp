#include "VanillaSIFT.h"
#include "logger.hpp"
using namespace cv::xfeatures2d;

// assumed gaussian blur for input image
const float VanillaSIFT::SIFT_INIT_SIGMA = 0.5f;

// determines gaussian sigma for orientation assignment
const float VanillaSIFT::SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
const float VanillaSIFT::SIFT_ORI_RADIUS = 3 * VanillaSIFT::SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
const float VanillaSIFT::SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
const float VanillaSIFT::SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor printf
const float VanillaSIFT::SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
const float VanillaSIFT::SIFT_INT_DESCR_FCTR = 512.f;

//------------------------------------VanillaSIFT()------------------------------------
// VanillaSIFT constructor, initialize variables
//Precondition: the following parameters must be correctly defined.
//parameters:
//nfeatures: 0 by default
//nOctaveLayers: 3 by default
//contrastThreshold: 0.04 by default
//edgeThreshold: 10 by default
//sigma: 1.6 by default
//Postcondition: variables are assigned
//-------------------------------------------------------------------------------------
VanillaSIFT::VanillaSIFT( int _nfeatures, int _nOctaveLayers, double _contrastThreshold, double _edgeThreshold, double _sigma )
    : nfeatures(_nfeatures), nOctaveLayers(_nOctaveLayers), contrastThreshold(_contrastThreshold), edgeThreshold(_edgeThreshold), sigma(_sigma)
{
}

//------------------------------------descriptorSize()---------------------------------
// ! returns the descriptor size in floats
//Precondition: None
//Postcondition: the descriptor size is returned in floats
//-------------------------------------------------------------------------------------
int VanillaSIFT::descriptorSize() const
{
    return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}

//------------------------------------descriptorType()---------------------------------
//! returns the descriptor type
//Precondition: None
//Postcondition: the descriptor type is returned
//-------------------------------------------------------------------------------------
int VanillaSIFT::descriptorType() const
{
    return CV_32F;
}

//------------------------------------operator()---------------------------------------
// Overloading operator() to run the SIFT algorithm :
// 1. compute keypoints using local extrema of Dog space
// 2. compute descriptors with keypoints
//Precondition: the following parameters must be correctly defined.
//parameters:
//img: color image base
//mask: image mask
//keypoints: keypoints of the image
//Postcondition: keypoints are assigned
//-------------------------------------------------------------------------------------
void VanillaSIFT::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& keypoints) const
{
	(*this)(_image, _mask, keypoints, noArray());
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
void VanillaSIFT::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& keypoints, OutputArray _descriptors, bool useProvidedKeypoints) const
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
			unpackOctave(keypoints[i], octave, layer, scale);
			firstOctave = std::min(firstOctave, octave);
			maxOctave = std::max(maxOctave, octave);
			actualNLayers = std::max(actualNLayers, layer - 2);
		}

		firstOctave = std::min(firstOctave, 0);
		CV_Assert(firstOctave >= -1 && actualNLayers <= nOctaveLayers);
		actualNOctaves = maxOctave - firstOctave + 1;
	}

	// base is a grey image
	Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);

	vector<Mat> gpyr, dogpyr;
	int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(log((double)std::min(base.cols, base.rows)) / log(2.) - 2) - firstOctave;

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
		KeyPointsFilter::runByPixelsMask( keypoints, mask );
	}

	if (_descriptors.needed())
	{
		//t = (double)getTickCount();
		int dsize = descriptorSize();
		_descriptors.create((int)keypoints.size(), dsize, CV_32F);
		Mat descriptors = _descriptors.getMat();

		calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
		//t = (double)getTickCount() - t;
		//printf("descriptor extraction time: %g\n", t*1000./tf);
	}
}

//------------------------------------detectImpl()-------------------------------------
//only detect keypoints without computing descriptors
//Precondition: the following parameters must be correctly defined.
//parameters:
//image: color image
//keypoints: empty vector to be filled
//mask: image mask
//Postcondition: keypoints are detected and assigned
//-------------------------------------------------------------------------------------
void VanillaSIFT::detectImpl(const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const 
{
	(*this)(image, mask, keypoints, noArray());
}

//------------------------------------computeImpl()------------------------------------
// call vanillaSIFT constructor and compute descriptors
//Precondition: the following parameters must be correctly defined.
//parameters:
//image: color image
//keypoints: empty vector to be filled
//descriptors: empty vector to be filled
//Postcondition: keypoints are detected and descriptors are assigned
//-------------------------------------------------------------------------------------
void VanillaSIFT::computeImpl(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const
{
	(*this)(image, Mat(), keypoints, descriptors, true);
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
void VanillaSIFT::compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors)
{
	this->computeImpl(image, keypoints, descriptors);
}

//------------------------------------createInitialImage()-----------------------------
//create initial grey-scale base image for later process
//Precondition: the following parameters must be correctly defined.
//parameters:
//img: color image
//doubleImageSize: bool indicating whether doubling the image 
//sigma: gaussian blur coefficient
//Postcondition: image is assigned
//-------------------------------------------------------------------------------------
Mat VanillaSIFT::createInitialImage(const Mat& img, bool doubleImageSize, float sigma) const {
    Logger::Log("createInitialImage - Start: doubleImageSize = " + std::string(doubleImageSize ? "true" : "false") + ", sigma = " + std::to_string(sigma));

    Mat gray, gray_fpt, output;
    if (img.channels() == 3 || img.channels() == 4) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
        Logger::Log("createInitialImage - Converted color image to grayscale.");
    } else {
        img.copyTo(gray);
        Logger::Log("createInitialImage - Image is already grayscale.");
    }
    gray.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);

    float sig_diff;
    if (doubleImageSize) {
        sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f));
        Mat dbl;
        resize(gray_fpt, dbl, Size(gray.cols * 2, gray.rows * 2), 0, 0, INTER_LINEAR);
        Logger::Log("createInitialImage - Image size doubled.");
        GaussianBlur(dbl, output, Size(), sig_diff, sig_diff);
    } else {
        sig_diff = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f));
        GaussianBlur(gray_fpt, output, Size(), sig_diff, sig_diff);
    }

    Logger::Log("createInitialImage - Gaussian blur applied with sigma diff = " + std::to_string(sig_diff));
    Logger::Log("createInitialImage - Resulting image dimensions: " + std::to_string(output.cols) + "x" + std::to_string(output.rows));

    return output;
}

//------------------------------------buildGaussianPyramid()---------------------------
// compute Gaussian pyramid using base image
//Precondition: the following parameters must be correctly defined.
//parameters:
//base: image base
//pyr: Mat vector to be assigned with gaussian blurred image
//nOctaves: number of octaves
//Postcondition: images are blurred and assigned to pyr
//-------------------------------------------------------------------------------------
void VanillaSIFT::buildGaussianPyramid(const Mat& base, std::vector<Mat>& pyr, int nOctaves) const {
    std::vector<double> sig(nOctaveLayers + 3);
    pyr.resize(nOctaves*(nOctaveLayers + 3));

    Logger::Log("buildGaussianPyramid - Starting pyramid construction. nOctaves: " + std::to_string(nOctaves) + ", nOctaveLayers: " + std::to_string(nOctaveLayers));

    // precompute Gaussian sigmas using the following formula:
    sig[0] = sigma;
    double k = std::pow(2., 1. / nOctaveLayers);
    for (int i = 1; i < nOctaveLayers + 3; i++) {
        double sig_prev = std::pow(k, (double)(i - 1)) * sigma;
        double sig_total = sig_prev * k;
        sig[i] = std::sqrt(sig_total * sig_total - sig_prev * sig_prev);
    }

    Logger::Log("buildGaussianPyramid - Sigma values computed for Gaussian blurring.");

    for (int o = 0; o < nOctaves; o++) {
        for (int i = 0; i < nOctaveLayers + 3; i++) {
            Mat& dst = pyr[o*(nOctaveLayers + 3) + i];
            if (o == 0 && i == 0) {
                dst = base;
                Logger::Log("buildGaussianPyramid - Base image assigned to the first layer of the pyramid.");
            } else if (i == 0) {
                const Mat& src = pyr[(o - 1)*(nOctaveLayers + 3) + nOctaveLayers];
                resize(src, dst, Size(src.cols / 2, src.rows / 2), 0, 0, INTER_NEAREST);
                Logger::Log("buildGaussianPyramid - Image resized for octave " + std::to_string(o) + ", layer " + std::to_string(i) + ".");
            } else {
                const Mat& src = pyr[o*(nOctaveLayers + 3) + i - 1];
                GaussianBlur(src, dst, Size(), sig[i], sig[i]);
                Logger::Log("buildGaussianPyramid - Gaussian blur applied for octave " + std::to_string(o) + ", layer " + std::to_string(i) + ", sigma: " + std::to_string(sig[i]) + ".");
            }
        }
    }

    Logger::Log("buildGaussianPyramid - Pyramid construction completed.");
}

//------------------------------------buildDoGPyramid()--------------------------------
// compute diffierence of Gaussian pyramid using Gaussian pyramid
//Precondition: the following parameters must be correctly defined.
//parameters:
//pyr: gaussian pyramid
//dogpyr: Mat array to be assigned with difference of Gaussian
//Postcondition: difference of Gaussian images are assigned to dogpyr
//-------------------------------------------------------------------------------------
#include "logger.hpp" // Ensure logger.hpp is included

void VanillaSIFT::buildDoGPyramid(const std::vector<Mat>& gpyr, std::vector<Mat>& dogpyr) const {
    int nOctaves = (int)gpyr.size() / (nOctaveLayers + 3);
    dogpyr.resize(nOctaves * (nOctaveLayers + 2));

    Logger::Log("buildDoGPyramid - Starting DoG pyramid construction. nOctaves: " + std::to_string(nOctaves));

    for (int o = 0; o < nOctaves; o++) {
        for (int i = 0; i < nOctaveLayers + 2; i++) {
            const Mat& src1 = gpyr[o * (nOctaveLayers + 3) + i];
            const Mat& src2 = gpyr[o * (nOctaveLayers + 3) + i + 1];
            Mat& dst = dogpyr[o * (nOctaveLayers + 2) + i];
            subtract(src2, src1, dst, noArray(), DataType<sift_wt>::type);

            Logger::Log("buildDoGPyramid - DoG image created for octave " + std::to_string(o) + ", layer " + std::to_string(i) +
                        ". Size: " + std::to_string(dst.cols) + "x" + std::to_string(dst.rows));
        }
    }

    Logger::Log("buildDoGPyramid - DoG pyramid construction completed.");
}


//------------------------------------calcOrientationHist()----------------------------
// Computes a gradient orientation histogram at a specified pixel
//Precondition: the following parameters must be correctly defined.
//parameters:
//img: color image
//pt: pixel location
//radius: histogram range
//sigma: 
//hist: orientation histogram
//n: newsift_descr_hist_bins, 8 in this case
//Postcondition: orientation is voted to histogram
//-------------------------------------------------------------------------------------
float VanillaSIFT::calcOrientationHist( const Mat& img, Point pt, int radius,
                                  float sigma, float* hist, int n )
{
    //Logger::Log("calcOrientationHist - start for pixel at position: " + std::to_string(pt.x) + ", " + std::to_string(pt.y));
    int i, j, k, len = (radius*2+1)*(radius*2+1);

    float expf_scale = -1.f/(2.f * sigma * sigma);
    AutoBuffer<float> buf(len*4 + n+4);
    float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
    float* temphist = W + len + 2;

    for( i = 0; i < n; i++ )
        temphist[i] = 0.f;

    for( i = -radius, k = 0; i <= radius; i++ )
    {
        int y = pt.y + i;
        if( y <= 0 || y >= img.rows - 1 )
            continue;
        for( j = -radius; j <= radius; j++ )
        {
            int x = pt.x + j;
            if( x <= 0 || x >= img.cols - 1 )
                continue;

            float dx = (float)(img.at<sift_wt>(y, x+1) - img.at<sift_wt>(y, x-1));
            float dy = (float)(img.at<sift_wt>(y-1, x) - img.at<sift_wt>(y+1, x));

            X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
            k++;
            // Log the calculated gradients and their positions
            //Logger::Log("Gradient calculation at (" + std::to_string(x) + ", " + std::to_string(y) +") - dx: " + std::to_string(dx) + ", dy: " + std::to_string(dy));
        }
    }

    len = k;

    // compute gradient values, orientations and the weights over the pixel neighborhood
    hal::exp(W, W, len);
    hal::fastAtan2(Y, X, Ori, len, true);
    hal::magnitude(X, Y, Mag, len);

    for( k = 0; k < len; k++ )
    {
        int bin = cvRound((n/360.f)*Ori[k]);
        if( bin >= n )
            bin -= n;
        if( bin < 0 )
            bin += n;
        temphist[bin] += W[k]*Mag[k];
    }

    // smooth the histogram
    temphist[-1] = temphist[n-1];
    temphist[-2] = temphist[n-2];
    temphist[n] = temphist[0];
    temphist[n+1] = temphist[1];
    for( i = 0; i < n; i++ )
    {
        hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f) +
            (temphist[i-1] + temphist[i+1])*(4.f/16.f) +
            temphist[i]*(6.f/16.f);
    }

    // Log the histogram after smoothing
//    Logger::Log("Orientation histogram after smoothing:");
//    for( i = 0; i < n; i++ ) {
//        Logger::Log("Bin[" + std::to_string(i) + "] = " + std::to_string(hist[i]));
//    }

    float maxval = hist[0];
    for( i = 1; i < n; i++ )
        maxval = std::max(maxval, hist[i]);

    // Log the maximum value of the histogram
    //Logger::Log("Maximum value in orientation histogram: " + std::to_string(maxval));

    //Logger::Log("calcOrientationHist - end for pixel at position: " + std::to_string(pt.x) + ", " + std::to_string(pt.y));

    return maxval;
}

//------------------------------------adjustLocalExtrema()-----------------------------
// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
//Precondition: the following parameters must be correctly defined.
//parameters:
//dog_pyr: difference of Gaussian
//kpt: pixel location
//octv: 
//layer: 
//r: row number
//c: column number
//nOctaveLayers: number of octave
//contrastThreshold:
//edgeThreshold:
//sigma:
//Postcondition: 
//-------------------------------------------------------------------------------------
bool VanillaSIFT::adjustLocalExtrema( const std::vector<Mat>& dog_pyr, KeyPoint& kpt, int octv, int& layer, int& r, int& c, int nOctaveLayers, float contrastThreshold, float edgeThreshold, float sigma )
{
    //Logger::Log("adjustLocalExtrema - start");
    const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
    const float deriv_scale = img_scale*0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale*0.25f;

    float xi=0, xr=0, xc=0, contr=0;
    int i = 0;

    for( ; i < SIFT_MAX_INTERP_STEPS; i++ )
    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];

        Vec3f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                 (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                 (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);

        float v2 = (float)img.at<sift_wt>(r, c)*2;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1))*cross_deriv_scale;
        float dxs = (next.at<sift_wt>(r, c+1) - next.at<sift_wt>(r, c-1) -
                     prev.at<sift_wt>(r, c+1) + prev.at<sift_wt>(r, c-1))*cross_deriv_scale;
        float dys = (next.at<sift_wt>(r+1, c) - next.at<sift_wt>(r-1, c) -
                     prev.at<sift_wt>(r+1, c) + prev.at<sift_wt>(r-1, c))*cross_deriv_scale;

        Matx33f H(dxx, dxy, dxs,
                  dxy, dyy, dys,
                  dxs, dys, dss);

        Vec3f X = H.solve(dD, DECOMP_LU);

        xi = -X[2];
        xr = -X[1];
        xc = -X[0];

        if( std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f ){
            //Logger::Log("Convergence reached at iteration: " + std::to_string(i));
            break;
        }

        if( std::abs(xi) > (float)(INT_MAX/3) ||
            std::abs(xr) > (float)(INT_MAX/3) ||
            std::abs(xc) > (float)(INT_MAX/3) )
            return false;

        c += cvRound(xc);
        r += cvRound(xr);
        layer += cvRound(xi);

        if( layer < 1 || layer > nOctaveLayers ||
            c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER  ||
            r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER )
            return false;
    }

    // ensure convergence of interpolation
    if( i >= SIFT_MAX_INTERP_STEPS ){
        //Logger::Log("Maximum interpolation steps reached without convergence.");
        return false;
    }


    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];
        Matx31f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                   (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                   (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);
        float t = dD.dot(Matx31f(xc, xr, xi));

        contr = img.at<sift_wt>(r, c)*img_scale + t * 0.5f;
        if( std::abs( contr ) * nOctaveLayers < contrastThreshold ){
            //Logger::Log("Contrast below threshold for keypoint.");
            return false;
        }

        // principal curvatures are computed using the trace and det of Hessian
        float v2 = img.at<sift_wt>(r, c)*2.f;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1)) * cross_deriv_scale;
        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det ){
            //Logger::Log("Edge response too strong and not a corner.");
            return false;
        }
    }

	kpt.pt.x = (c + xc) * (1 << octv);
	kpt.pt.y = (r + xr) * (1 << octv);
	kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5) * 255) << 16);
	kpt.size = sigma * powf(2.f, (layer + xi) / nOctaveLayers) * (1 << octv) * 2;
	kpt.response = std::abs(contr);

    //Logger::Log("Keypoint adjusted and accepted - Position: (" + std::to_string(kpt.pt.x) + ", " + std::to_string(kpt.pt.y) + ")");
    return true;
}

//------------------------------------findScaleSpaceExtrema()--------------------------
// Detects features at extrema in DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.
//Precondition: the following parameters must be correctly defined.
//parameters:
//gauss_pyr: gaussian pyramid
//dog_pyr: difference of Gaussian pyramid
//keypoints: empty keypoints vector
//Postcondition: keypoints are assigned
//-------------------------------------------------------------------------------------
void VanillaSIFT::findScaleSpaceExtrema( const std::vector<Mat>& gauss_pyr, const std::vector<Mat>& dog_pyr,
                                  std::vector<KeyPoint>& keypoints ) const
{
    //Logger::Log("findScaleSpaceExtrema - start");


    int nOctaves = (int)gauss_pyr.size()/(nOctaveLayers + 3);
    int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);
    const int n = SIFT_ORI_HIST_BINS;
    float hist[n];
    KeyPoint kpt;

    keypoints.clear();

    for( int o = 0; o < nOctaves; o++ )
        for( int i = 1; i <= nOctaveLayers; i++ )
        {
            int idx = o*(nOctaveLayers+2)+i;
            const Mat& img = dog_pyr[idx];
            const Mat& prev = dog_pyr[idx-1];
            const Mat& next = dog_pyr[idx+1];
            int step = (int)img.step1();
            int rows = img.rows, cols = img.cols;

            for( int r = SIFT_IMG_BORDER; r < rows-SIFT_IMG_BORDER; r++)
            {
                const sift_wt* currptr = img.ptr<sift_wt>(r);
                const sift_wt* prevptr = prev.ptr<sift_wt>(r);
                const sift_wt* nextptr = next.ptr<sift_wt>(r);

                for( int c = SIFT_IMG_BORDER; c < cols-SIFT_IMG_BORDER; c++)
                {
                    sift_wt val = currptr[c];

                    // find local extrema with pixel accuracy
                    if( std::abs(val) > threshold &&
                       ((val > 0 && val >= currptr[c-1] && val >= currptr[c+1] &&
                         val >= currptr[c-step-1] && val >= currptr[c-step] && val >= currptr[c-step+1] &&
                         val >= currptr[c+step-1] && val >= currptr[c+step] && val >= currptr[c+step+1] &&
                         val >= nextptr[c] && val >= nextptr[c-1] && val >= nextptr[c+1] &&
                         val >= nextptr[c-step-1] && val >= nextptr[c-step] && val >= nextptr[c-step+1] &&
                         val >= nextptr[c+step-1] && val >= nextptr[c+step] && val >= nextptr[c+step+1] &&
                         val >= prevptr[c] && val >= prevptr[c-1] && val >= prevptr[c+1] &&
                         val >= prevptr[c-step-1] && val >= prevptr[c-step] && val >= prevptr[c-step+1] &&
                         val >= prevptr[c+step-1] && val >= prevptr[c+step] && val >= prevptr[c+step+1]) ||
                        (val < 0 && val <= currptr[c-1] && val <= currptr[c+1] &&
                         val <= currptr[c-step-1] && val <= currptr[c-step] && val <= currptr[c-step+1] &&
                         val <= currptr[c+step-1] && val <= currptr[c+step] && val <= currptr[c+step+1] &&
                         val <= nextptr[c] && val <= nextptr[c-1] && val <= nextptr[c+1] &&
                         val <= nextptr[c-step-1] && val <= nextptr[c-step] && val <= nextptr[c-step+1] &&
                         val <= nextptr[c+step-1] && val <= nextptr[c+step] && val <= nextptr[c+step+1] &&
                         val <= prevptr[c] && val <= prevptr[c-1] && val <= prevptr[c+1] &&
                         val <= prevptr[c-step-1] && val <= prevptr[c-step] && val <= prevptr[c-step+1] &&
                         val <= prevptr[c+step-1] && val <= prevptr[c+step] && val <= prevptr[c+step+1])))
                    {
                        int r1 = r, c1 = c, layer = i;
                        if( !adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
                                                nOctaveLayers, (float)contrastThreshold,
                                                (float)edgeThreshold, (float)sigma) )
                            continue;
                        float scl_octv = kpt.size*0.5f/(1 << o);
                        float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers+3) + layer],
                                                         Point(c1, r1),
                                                         cvRound(SIFT_ORI_RADIUS * scl_octv),
                                                         SIFT_ORI_SIG_FCTR * scl_octv,
                                                         hist, n);
                        float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
                        for( int j = 0; j < n; j++ )
                        {
                            int l = j > 0 ? j - 1 : n - 1;
                            int r2 = j < n-1 ? j + 1 : 0;

                            if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
                            {
                                //Logger::Log("Keypoint found - octave: " + std::to_string(o) + " layer: " + std::to_string(i));
                                float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
                                bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
                                kpt.angle = 360.f - (float)((360.f/n) * bin);
                                if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
                                    kpt.angle = 0.f;
                                keypoints.push_back(kpt);
                            }
                        }
                    }
                }
            }
        }

    //Logger::Log("findScaleSpaceExtrema - end");
}

//------------------------------------calcSIFTDescriptor()-----------------------------
// compute SIFT descriptor and perform normalization for one keypoint
//Precondition: the following parameters must be correctly defined.
//parameters:
//img: color image
//ptf: keypoint
//ori: angle(degree) of the keypoint relative to the coordinates, clockwise
//scl: radius of meaningful neighborhood around the keypoint 
//d: newsift descr_width, 4 in this case
//n: newsift_descr_hist_bins, 8 in this case
//dst: descriptor array to pass in
//Postcondition: descriptors are assigned to dst
//-------------------------------------------------------------------------------------
void VanillaSIFT::calcSIFTDescriptor( const Mat& img, Point2f ptf, float ori, float scl,
                               int d, int n, float* dst ) const
{
    Logger::Log("calcSIFTDescriptor - start for keypoint at position: " + std::to_string(ptf.x) + ", " + std::to_string(ptf.y));
    Point pt(cvRound(ptf.x), cvRound(ptf.y));
    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = std::min(radius, (int) sqrt((double) img.cols*img.cols + img.rows*img.rows));
    cos_t /= hist_width;
    sin_t /= hist_width;

    int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (d+2)*(d+2)*(n+2);
    int rows = img.rows, cols = img.cols;

    AutoBuffer<float> buf(len*6 + histlen);
    float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
    float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

    for( i = 0; i < d+2; i++ )
    {
        for( j = 0; j < d+2; j++ )
            for( k = 0; k < n+2; k++ )
                hist[(i*(d+2) + j)*(n+2) + k] = 0.;
    }

    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = pt.y + i, c = pt.x + j;

            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {
                float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
                float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
                X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
                W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
                k++;
                //Logger::Log("calcSIFTDescriptor - gradient magnitude and orientation calculated for sample at position: " + std::to_string(r) + ", " + std::to_string(c));
            }
        }

    len = k;
    hal::fastAtan2(Y, X, Ori, len, true);
    hal::magnitude(X, Y, Mag, len);
    hal::exp(W, W, len);

    for( k = 0; k < len; k++ )
    {
        float rbin = RBin[k], cbin = CBin[k];
        float obin = (Ori[k] - ori)*bins_per_rad;
        float mag = Mag[k]*W[k];

        int r0 = cvFloor( rbin );
        int c0 = cvFloor( cbin );
        int o0 = cvFloor( obin );
        rbin -= r0;
        cbin -= c0;
        obin -= o0;

        if( o0 < 0 )
            o0 += n;
        if( o0 >= n )
            o0 -= n;

        // histogram update using tri-linear interpolation
        float v_r1 = mag*rbin, v_r0 = mag - v_r1;
        float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
        float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
        float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
        float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
        float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
        float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

        int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
        hist[idx] += v_rco000;
        hist[idx+1] += v_rco001;
        hist[idx+(n+2)] += v_rco010;
        hist[idx+(n+3)] += v_rco011;
        hist[idx+(d+2)*(n+2)] += v_rco100;
        hist[idx+(d+2)*(n+2)+1] += v_rco101;
        hist[idx+(d+3)*(n+2)] += v_rco110;
        hist[idx+(d+3)*(n+2)+1] += v_rco111;
    }

    // finalize histogram, since the orientation histograms are circular
    Logger::Log("calcSIFTDescriptor - histogram before normalization:");
    int nonZeroBins = 0;
    float maxBinValue = 0;
    for (i = 0; i < d * d * n; i++) {
        if (dst[i] > 0) nonZeroBins++;
        maxBinValue = std::max(maxBinValue, dst[i]);
    }
    Logger::Log("Histogram summary: non-zero bins = " + std::to_string(nonZeroBins) + ", max bin value = " + std::to_string(maxBinValue));
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    len = d*d*n;
    Logger::Log("calcSIFTDescriptor - histogram after normalization:");
    for( k = 0; k < len; k++ ){
        //Logger::Log("Pre-Normalization Bin[" + std::to_string(k) + "] = " + std::to_string(dst[k]));
        nrm2 += dst[k]*dst[k];
    }

    float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;

    int thresholdedCount = 0; // Count how many bins were thresholded
    float maxBinValueAfterThresholding = 0; // Maximum bin value after thresholding
    float sumBinValues = 0; // Sum of all bin values after thresholding

    for(i = 0, nrm2 = 0; i < k; i++) {
        float originalVal = dst[i];
        float val = std::min(dst[i], thr);
        dst[i] = val;
        nrm2 += val * val;

        // Update statistics
        if (val != originalVal) thresholdedCount++;
        maxBinValueAfterThresholding = std::max(maxBinValueAfterThresholding, val);
        sumBinValues += val;
    }

// Now log the summary
    Logger::Log("Thresholding summary: Thresholded bins = " + std::to_string(thresholdedCount) +
                ", Max bin value after thresholding = " + std::to_string(maxBinValueAfterThresholding) +
                ", Sum of bin values = " + std::to_string(sumBinValues) +
                ", Normalization factor = " + std::to_string(nrm2));

    nrm2 = SIFT_INT_DESCR_FCTR/std::max(std::sqrt(nrm2), FLT_EPSILON);

    maxBinValue = 0; // Maximum bin value after normalization
    float minBinValue = std::numeric_limits<float>::max(); // Minimum bin value, initialized to the largest possible float value
    sumBinValues = 0; // Sum of all bin values for calculating average

    for(k = 0; k < len; k++) {
        dst[k] = dst[k] * nrm2; // Normalization applied here
        maxBinValue = std::max(maxBinValue, dst[k]);
        minBinValue = std::min(minBinValue, dst[k]);
        sumBinValues += dst[k];
    }

    float averageBinValue = len > 0 ? sumBinValues / len : 0; // Calculate the average bin value

// Log the summary of normalization
    Logger::Log("Normalization summary: Max bin value = " + std::to_string(maxBinValue) +
                ", Min bin value = " + std::to_string(minBinValue) +
                ", Average bin value = " + std::to_string(averageBinValue));
    Logger::Log("calcSIFTDescriptor - end for keypoint at position: " + std::to_string(ptf.x) + ", " + std::to_string(ptf.y));
}

//------------------------------------calcDescriptors()--------------------------------
// set up variables and call calcSIFTDescriptor() to compute descriptors
//Precondition: the following parameters must be correctly defined.
//parameters:
//gpyr: gaussian pyramid
//keypoints: computed keypoints
//descriptors: empty descriptors vector
//nOctaveLayers: number of octave layers
//firstOctave: index of first octave
//Postcondition: descriptors are assigned
//-------------------------------------------------------------------------------------
void VanillaSIFT::calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints,
                                  Mat& descriptors, int nOctaveLayers, int firstOctave ) const
{
    int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

    Logger::Log("calcDescriptors - Starting descriptor calculation for " + std::to_string(keypoints.size()) + " keypoints.");

    for( size_t i = 0; i < keypoints.size(); i++ )
    {
        KeyPoint kpt = keypoints[i];
        int octave, layer;
        float scale;
        unpackOctave(kpt, octave, layer, scale);
        CV_Assert(octave >= firstOctave && layer <= nOctaveLayers+2);
        float size = kpt.size * scale;
        Point2f ptf(kpt.pt.x * scale, kpt.pt.y * scale);
        const Mat& img = gpyr[(octave - firstOctave) * (nOctaveLayers + 3) + layer];

        float angle = 360.f - kpt.angle;
        if(std::abs(angle - 360.f) < FLT_EPSILON)
            angle = 0.f;

        Logger::Log("Processing keypoint " + std::to_string(i + 1) + " / " + std::to_string(keypoints.size()) +
                    ": Octave = " + std::to_string(octave) +
                    ", Layer = " + std::to_string(layer) +
                    ", Position = (" + std::to_string(ptf.x) + ", " + std::to_string(ptf.y) +
                    "), Scale = " + std::to_string(scale) +
                    ", Size = " + std::to_string(size) +
                    ", Angle = " + std::to_string(angle));

        calcSIFTDescriptor(img, ptf, angle, size * 0.5f, d, n, descriptors.ptr<float>((int)i));
    }

    Logger::Log("calcDescriptors - Descriptor calculation completed.");
}

