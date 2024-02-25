#include "descriptorProcessor.hpp"

// Processes image descriptors based on specified options.
cv::Mat DescriptorProcessor::processDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
                                                const cv::Ptr<cv::Feature2D>& featureExtractor,
                                                const DescriptorOptions& options) {

    // if the descriptor is a grayscale type and the image is not, convert it to grayscale
    if(options.descriptorColorSpace == D_COLOR) {
        if (image.channels() == 3) {
            // Image is in BGR format, convert it to grayscale
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        } else if (image.channels() == 4) {
            // For example, convert BGRA to grayscale
            cv::cvtColor(image, image, cv::COLOR_BGRA2GRAY);
        }
    }

    cv::Mat descriptors;

// TODO: remove if we decide to not use custom descriptor types
//
//    if (options.descriptorType == DESCRIPTOR_VSIFT) {
//
//        // allocate a pointer to a VanillaSIFT object
//        Ptr<VanillaSIFT> vanillaSiftExtractor = VanillaSIFT::create();
//
//        if (vanillaSiftExtractor) {
//            // Successful cast to VanillaSIFT, proceed with specific operations
//            if (options.poolingStrategy == DOMAIN_SIZE_POOLING) {
//                descriptors = sumPooling(image, keypoints, vanillaSiftExtractor, options.scales, options);
//            } else if (options.poolingStrategy == AVERAGE_POOLING) {
//                descriptors = averagePooling(image, keypoints, vanillaSiftExtractor, options.scales, options);
//            } else if (options.poolingStrategy == MAX_POOLING) {
//                descriptors = maxPooling(image, keypoints, vanillaSiftExtractor, options.scales, options);
//            } else {
//                vanillaSiftExtractor->compute(image, keypoints, descriptors);
//            }
//        } else {
//            throw std::runtime_error("Feature extractor is not a VanillaSIFT instance.");
//        }
//    } else {
//        // Fallback or generic handling using featureExtractor as a generic cv::Feature2D
//        // This path will use the base class's compute method if the specific cast fails.
//        if (options.poolingStrategy == DOMAIN_SIZE_POOLING) {
//            descriptors = sumPooling(image, keypoints, featureExtractor, options.scales, options);
//        } else if (options.poolingStrategy == AVERAGE_POOLING) {
//            descriptors = averagePooling(image, keypoints, featureExtractor, options.scales, options);
//        } else if (options.poolingStrategy == MAX_POOLING) {
//            descriptors = maxPooling(image, keypoints, featureExtractor, options.scales, options);
//        } else {
//            featureExtractor->compute(image, keypoints, descriptors);
//        }
//    }

    // Pooling strategies
    if (options.poolingStrategy == DOMAIN_SIZE_POOLING) {
        descriptors = sumPooling(image, keypoints, featureExtractor, options.scales, options);
    }else if (options.poolingStrategy == AVERAGE_POOLING) {
        descriptors = averagePooling(image, keypoints, featureExtractor, options.scales, options);
    } else if (options.poolingStrategy == MAX_POOLING) {
        descriptors = maxPooling(image, keypoints, featureExtractor, options.scales, options);
    } else if (options.poolingStrategy == STACKING) {
        descriptors = stackDescriptors(image, keypoints, featureExtractor, options.scales, options);
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
cv::Mat DescriptorProcessor::sumPooling(const cv::Mat& image,
                                        const std::vector<cv::KeyPoint>& keypoints,
                                        const cv::Ptr<cv::Feature2D>& featureExtractor,
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
                                            const cv::Ptr<cv::Feature2D>& featureExtractor, const std::vector<float> &scales,
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
                                        const cv::Ptr<cv::Feature2D>& featureExtractor, const std::vector<float>& scales,
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

cv::Mat DescriptorProcessor::stackDescriptors(const cv::Mat& image,
                                              const std::vector<cv::KeyPoint>& keypoints,
                                              const cv::Ptr<cv::Feature2D>& featureExtractor,
                                              const std::vector<float>& scales,
                                              const DescriptorOptions& options) {
    cv::Mat stackedDescriptors;
    for (auto scale : scales) {
        cv::Mat im_scaled;
        cv::resize(image, im_scaled, cv::Size(), scale, scale);
        std::vector<cv::KeyPoint> keypoints_scaled;
        for (auto kp : keypoints) {
            keypoints_scaled.emplace_back(kp.pt * scale, kp.size * scale);
        }
        cv::Mat descriptors_scaled;
        featureExtractor->compute(im_scaled, keypoints_scaled, descriptors_scaled);

        if (options.normalizationStage == BEFORE_POOLING) {
            cv::normalize(descriptors_scaled, descriptors_scaled, 1, 0, options.normType);
        }

        if (options.rootingStage == R_BEFORE_POOLING) {
            rootDescriptors(descriptors_scaled);
        }

        if (stackedDescriptors.empty()) {
            stackedDescriptors = descriptors_scaled;
        } else {
            cv::vconcat(stackedDescriptors, descriptors_scaled, stackedDescriptors);
        }
    }

    // Optionally, you can normalize the stackedDescriptors here again

    return stackedDescriptors;
}



    // Method to apply square root to descriptors
void DescriptorProcessor::rootDescriptors(cv::Mat& descriptors) {
    for (int i = 0; i < descriptors.rows; ++i) {
        for (int j = 0; j < descriptors.cols; ++j) {
            descriptors.at<float>(i, j) = std::sqrt(descriptors.at<float>(i, j));
        }
    }
}


