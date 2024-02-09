#include "descriptorProcessor.hpp"

// Processes image descriptors based on specified options.
cv::Mat DescriptorProcessor::processDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
                                                cv::Ptr<cv::Feature2D> featureExtractor,
                                                const DescriptorOptions& options) {
    cv::Mat descriptors;

    // Pooling strategies
    if (options.poolingStrategy == DOMAIN_SIZE_POOLING) {
        descriptors = computeDescriptorsWithScaling(image, keypoints, featureExtractor, options.scales, options);
    }else if (options.poolingStrategy == AVERAGE_POOLING) {
        descriptors = averagePooling(image, keypoints, featureExtractor, options.scales, options);
    } else if (options.poolingStrategy == MAX_POOLING) {
        descriptors = maxPooling(image, keypoints, featureExtractor, options.scales, options);
    }else {
        featureExtractor->compute(image, keypoints, descriptors);
    }

    // Apply normalization after pooling
    if (options.normalizationStage == AFTER_POOLING) {
        cv::normalize(descriptors, descriptors, 1, 0, options.normType);
    }

    // Apply rooting after pooling if specified
    if (options.rootingStage == R_AFTER_POOLING) {
        rootDescriptors(descriptors);
    }

    return descriptors;
}

// Compute descriptors with scaling
cv::Mat DescriptorProcessor::computeDescriptorsWithScaling(const cv::Mat& image,
                                                           const std::vector<cv::KeyPoint>& keypoints,
                                                           cv::Ptr<cv::Feature2D> featureExtractor,
                                                           const std::vector<float>& scales,
                                                           const DescriptorOptions& options) {
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

        if(options.normalizationStage == BEFORE_POOLING) {
            cv::normalize(descriptors_scaled, descriptors_scaled, 1, 0, options.normType);
        }

        if(options.rootingStage == R_BEFORE_POOLING) {
            rootDescriptors(descriptors_scaled);
        }

        if (sumOfDescriptors.empty()) {
            sumOfDescriptors = cv::Mat::zeros(descriptors_scaled.rows, descriptors_scaled.cols,
                                              descriptors_scaled.type());
        }
        sumOfDescriptors += descriptors_scaled;
    }

    // Optionally, you can normalize the sumOfDescriptors here again

    return sumOfDescriptors;
}

cv::Mat DescriptorProcessor::averagePooling(const cv::Mat &image, const std::vector<cv::KeyPoint> &keypoints,
                                            cv::Ptr<cv::Feature2D> featureExtractor, const std::vector<float> &scales,
                                            const DescriptorOptions& options) {
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

        if(options.normalizationStage == BEFORE_POOLING) {
            cv::normalize(descriptors_scaled, descriptors_scaled, 1, 0, options.normType);
        }

        if(options.rootingStage == R_BEFORE_POOLING) {
            rootDescriptors(descriptors_scaled);
        }

        if (sumOfDescriptors.empty()) {
            sumOfDescriptors = cv::Mat::zeros(descriptors_scaled.rows, descriptors_scaled.cols,
                                              descriptors_scaled.type());
        }
        sumOfDescriptors += descriptors_scaled;
    }

    // average all the indexes in the sumOfDescriptors
    for (int i = 0; i < sumOfDescriptors.rows; ++i) {
        for (int j = 0; j < sumOfDescriptors.cols; ++j) {
            sumOfDescriptors.at<float>(i, j) = sumOfDescriptors.at<float>(i, j) / scales.size();
        }

    }

    return sumOfDescriptors;
}

// Compute descriptors with max pooling
cv::Mat DescriptorProcessor::maxPooling(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints,
                                        cv::Ptr<cv::Feature2D> featureExtractor, const std::vector<float>& scales,
                                        const DescriptorOptions& options) {
    cv::Mat maxOfDescriptors;
    for (auto scale : scales) {
        cv::Mat im_scaled;
        cv::resize(image, im_scaled, cv::Size(), scale, scale);
        std::vector<cv::KeyPoint> keypoints_scaled;
        for (auto kp : keypoints) {
            keypoints_scaled.emplace_back(kp.pt * scale, kp.size * scale);
        }
        cv::Mat descriptors_scaled;
        featureExtractor->compute(im_scaled, keypoints_scaled, descriptors_scaled);

        if(options.normalizationStage == BEFORE_POOLING) {
            cv::normalize(descriptors_scaled, descriptors_scaled, 1, 0, options.normType);
        }

        if(options.rootingStage == R_BEFORE_POOLING) {
            rootDescriptors(descriptors_scaled);
        }

        if (maxOfDescriptors.empty()) {
            maxOfDescriptors = descriptors_scaled.clone();
        } else {
            cv::max(maxOfDescriptors, descriptors_scaled, maxOfDescriptors);
        }
    }

    // No need to normalize here again unless it's part of your experiment's specific requirements

    return maxOfDescriptors;
}


    // Method to apply square root to descriptors
void DescriptorProcessor::rootDescriptors(cv::Mat& descriptors) {
    for (int i = 0; i < descriptors.rows; ++i) {
        for (int j = 0; j < descriptors.cols; ++j) {
            descriptors.at<float>(i, j) = std::sqrt(descriptors.at<float>(i, j));
        }
    }
}


