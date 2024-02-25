#ifndef DESCRIPTOR_DESCRIPTORPROCESSOR_HPP
#define DESCRIPTOR_DESCRIPTORPROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <memory>
//#include "../keypoints/HoNC.h"
//#include "../keypoints/VanillaSIFT.h"

// Enum for different pooling strategies
enum PoolingStrategy {
    NONE, // No pooling
    AVERAGE_POOLING,
    MAX_POOLING,
    DOMAIN_SIZE_POOLING // Domain size pooling is selected here
};

// Enum for when to apply normalization
enum NormalizationStage {
    BEFORE_POOLING,
    AFTER_POOLING,
    NO_NORMALIZATION // Skip normalization
};

// enum for rooting stage
enum RootingStage {
    R_BEFORE_POOLING,
    R_AFTER_POOLING,
    R_NONE
};

enum DescriptorType {
    DESCRIPTOR_SIFT,
    DESCRIPTOR_ORB,
    DESCRIPTOR_SURF
    // Add more descriptor types as needed
};

enum DescriptorColorSpace {
    D_COLOR,
    D_BW
};

enum ImageType {
    COLOR,
    BW
};

struct DescriptorOptions {
    PoolingStrategy poolingStrategy = NONE;
    std::vector<float> scales = {1.0f}; // Used for domain size pooling
    int normType = cv::NORM_L1;
    NormalizationStage normalizationStage = NO_NORMALIZATION;
    RootingStage rootingStage = R_AFTER_POOLING;
    ImageType imageType = BW;
    DescriptorType descriptorType = DESCRIPTOR_SIFT;
    DescriptorColorSpace descriptorColorSpace = D_BW;

    // constructor
    DescriptorOptions() = default;
};


/**
 * @class DescriptorProcessor
 * @brief Processes image descriptors using specified options and feature extractors.
 *
 * This class provides static methods to process image descriptors according to
 * given DescriptorOptions. It supports operations like scaling, normalization,
 * and rooting of descriptors. The class is designed to work with any OpenCV
 * feature extractor passed as a cv::Ptr<cv::Feature2D>.
 */
class DescriptorProcessor {
public:
    /**
     * Processes image descriptors based on specified options.
     *
     * @param image The image to process.
     * @param keypoints A vector of keypoints where descriptors should be computed.
     * @param featureExtractor A pointer to the feature extractor (e.g., SIFT, SURF).
     * @param options The DescriptorOptions struct specifying how descriptors should be processed.
     * @return A cv::Mat containing the processed descriptors.
     */
    static cv::Mat processDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
                                      const cv::Ptr<cv::Feature2D>& featureExtractor, const DescriptorOptions& options);

private:
    /**
     * Computes descriptors at multiple scales, if scaling is enabled.
     *
     * @param image The image to process.
     * @param keypoints The keypoints where descriptors should be computed.
     * @param featureExtractor The feature extractor to use.
     * @param scales The scales at which to compute the descriptors.
     * @return A cv::Mat containing descriptors computed at multiple scales.
     */
    static cv::Mat sumPooling(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints,
                              const cv::Ptr<cv::Feature2D>& featureExtractor,
                              const std::vector<float>& scales, const DescriptorOptions& options);
    static cv::Mat averagePooling(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints,
                                  const cv::Ptr<cv::Feature2D>& featureExtractor, const std::vector<float>& scales,
                                  const DescriptorOptions& options);
    static cv::Mat maxPooling(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints,
                              const cv::Ptr<cv::Feature2D>& featureExtractor, const std::vector<float>& scales,
                              const DescriptorOptions& options);

    /**
     * Applies square rooting to each element of the descriptorExtractor matrix.
     *
     * @param descriptors The descriptorExtractor matrix to be rooted.
     */
    static void rootDescriptors(cv::Mat& descriptors);
};


#endif //DESCRIPTOR_DESCRIPTORPROCESSOR_HPP
