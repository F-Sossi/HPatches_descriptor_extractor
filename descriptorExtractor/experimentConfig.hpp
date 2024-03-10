#ifndef EXPERIMENT_CONFIG_HPP
#define EXPERIMENT_CONFIG_HPP

#include <utility>

#include "descriptorProcessor.hpp" // Header with options structure

/**
 * @struct ExperimentConfig
 * @brief Encapsulates configuration options for descriptor processing experiments.
 *
 * This structure holds all necessary configuration options that affect
 * descriptor processing, allowing for easy modification of experiments.
 */
struct ExperimentConfig {
    DescriptorOptions descriptorOptions;
    DescriptorType descriptorType; // Added field to store descriptor type
    bool useMultiThreading = true; // Added field to enable multi-threading

    // Constructor
    ExperimentConfig() : descriptorType(DESCRIPTOR_SIFT) {}

    // Initialize with specific descriptor options and type
    ExperimentConfig(DescriptorOptions options, DescriptorType type)
            : descriptorOptions(std::move(options)), descriptorType(type) {
        // set options.descriptorType to the type
        descriptorOptions.descriptorType = type;
    }

    void setDescriptorOptions(const DescriptorOptions& options) {
        descriptorOptions = options;
    }

    void setDescriptorType(DescriptorType type) {
        descriptorType = type;
    }

    void configurePoolingStrategy(PoolingStrategy strategy, const std::vector<float>& scales) {
        descriptorOptions.poolingStrategy = strategy;
        descriptorOptions.scales = scales;
    }

    void configureNormalization(int normType, NormalizationStage normStage) {
        descriptorOptions.normType = normType;
        descriptorOptions.normalizationStage = normStage;
    }

    void configureRooting(RootingStage rootingStage) {
        descriptorOptions.rootingStage = rootingStage;
    }

    // Method to create the descriptor extractor based on the selected type
    [[nodiscard]] cv::Ptr<cv::Feature2D> createDescriptorExtractor() const {
        switch(descriptorType) {
            case DESCRIPTOR_SIFT:
                return cv::SIFT::create();
            case DESCRIPTOR_SURF:
                return cv::xfeatures2d::SURF::create();
            default:
                // Print error message and return SIFT as default
                std::cerr << "Unknown descriptor type, using SIFT as default" << std::endl;
                exit(1);
                //return cv::SIFT::create(); // Default case
        }
    }
};

#endif // EXPERIMENT_CONFIG_HPP

