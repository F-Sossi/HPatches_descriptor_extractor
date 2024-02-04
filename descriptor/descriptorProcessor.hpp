//
// Created by frank on 2/3/24.
//

#ifndef DESCRIPTOR_DESCRIPTORPROCESSOR_HPP
#define DESCRIPTOR_DESCRIPTORPROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp> // For SURF, if available
#include <vector>
#include <memory>

// Option structure for descriptor modifications
struct DescriptorOptions {
    bool useScaling = false;
    std::vector<float> scales = {1.0f}; // Default to no scaling
    bool normalize = false;
    int normType = cv::NORM_L1; // Default normalization type
    bool useRooting = false;
    // Optional: Include a field for the descriptor type or the feature extractor itself
    // std::string descriptorType = "SIFT"; // Example: Could also be "SURF", "ORB", etc.
};

class DescriptorProcessor {
public:
    // Method to process descriptors with given options
    // Accept a pointer to the feature extractor as a parameter
    static cv::Mat processDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Ptr<cv::Feature2D> featureExtractor, DescriptorOptions options) {
        cv::Mat descriptors;
        if (options.useScaling) {
            descriptors = computeDescriptorsWithScaling(image, keypoints, featureExtractor, options.scales);
        } else {
            featureExtractor->compute(image, keypoints, descriptors);
        }

        if (options.normalize) {
            cv::normalize(descriptors, descriptors, 1, 0, options.normType);
        }

        if (options.useRooting) {
            rootDescriptors(descriptors);
        }

        return descriptors;
    }

private:
    // Compute descriptors with scaling
    static cv::Mat computeDescriptorsWithScaling(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, cv::Ptr<cv::Feature2D> featureExtractor, const std::vector<float>& scales) {
        cv::Mat sumOfDescriptors;
        for (auto scale : scales) {
            cv::Mat im_scaled;
            cv::resize(image, im_scaled, cv::Size(), scale, scale);
            std::vector<cv::KeyPoint> keypoints_scaled;
            for (auto kp : keypoints) {
                keypoints_scaled.emplace_back(kp.pt * scale, kp.size * scale);
            }
            cv::Mat descriptors_scaled;
            featureExtractor->compute(im_scaled, keypoints_scaled, descriptors_scaled);

            if (sumOfDescriptors.empty()) {
                sumOfDescriptors = cv::Mat::zeros(descriptors_scaled.rows, descriptors_scaled.cols, descriptors_scaled.type());
            }
            sumOfDescriptors += descriptors_scaled;
        }

        // Optionally, you can normalize the sumOfDescriptors here again

        return sumOfDescriptors;
    }

    // Method to apply square root to descriptors
    static void rootDescriptors(cv::Mat& descriptors) {
        for (int i = 0; i < descriptors.rows; ++i) {
            for (int j = 0; j < descriptors.cols; ++j) {
                descriptors.at<float>(i, j) = std::sqrt(descriptors.at<float>(i, j));
            }
        }
    }
};


#endif //DESCRIPTOR_DESCRIPTORPROCESSOR_HPP
