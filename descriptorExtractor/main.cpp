#include "experimentConfig.hpp"
#include "hpatchesDescriptorExtractor.hpp"
#include <vector>
#include <opencv2/core.hpp> // For cv::NORM_L1, etc.
#include <string>

// Helper function to convert enum and settings to string for naming
std::string poolingStrategyToString(PoolingStrategy strategy) {
    switch (strategy) {
        case NONE: return "None";
        case AVERAGE_POOLING: return "Avg";
        case MAX_POOLING: return "Max";
        case DOMAIN_SIZE_POOLING: return "Dom";
        default: return "UnkPool";
    }
}

std::string normalizationStageToString(NormalizationStage stage) {
    switch (stage) {
        case BEFORE_POOLING: return "Bef";
        case AFTER_POOLING: return "Aft";
        case NO_NORMALIZATION: return "NoNorm";
        default: return "UnkNorm";
    }
}

std::string rootingStageToString(RootingStage stage) {
    switch (stage) {
        case R_BEFORE_POOLING: return "RBef";
        case R_AFTER_POOLING: return "RAft";
        default: return "UnkRoot";
    }
}

std::string normTypeToString(int normType) {
    switch (normType) {
        case cv::NORM_L1: return "L1";
        case cv::NORM_L2: return "L2";
        default: return "UnkNormType";
    }
}


int main() {
    // Define all possible options
    std::vector<PoolingStrategy> poolingStrategies = {AVERAGE_POOLING, MAX_POOLING, DOMAIN_SIZE_POOLING};
    std::vector<NormalizationStage> normalizationStages = {BEFORE_POOLING, AFTER_POOLING, NO_NORMALIZATION};
    std::vector<RootingStage> rootingStages = {R_BEFORE_POOLING};
    std::vector<int> normTypes = {cv::NORM_L1}; // Add more norm types if needed

    std::string directoryPath = "../data";

    // Iterate over all combinations of options
    for (auto& pooling : poolingStrategies) {
        for (auto& normalization : normalizationStages) {
            for (auto& rooting : rootingStages) {
                for (auto& normType : normTypes) {
                    // Configure the descriptor options
                    DescriptorOptions options;
                    options.poolingStrategy = pooling;
                    options.scales = {1.0f, 1.5f, 2.0}; // Example scales, modify as needed
                    options.normType = normType;
                    options.normalizationStage = normalization;
                    options.rootingStage = rooting;

                    // Create experiment configuration
                    ExperimentConfig config(options);

                    // Create a descriptive experiment name
                    std::string descriptorName = "Sift3up" + poolingStrategyToString(pooling) +
                                                 normalizationStageToString(normalization) +
                                                 rootingStageToString(rooting) +
                                                 normTypeToString(normType);

                    // Run the descriptor extraction process
                    hpatchesDescriptorExtractor::processImages(descriptorName, directoryPath, config);
                }
            }
        }
    }

    return 0;
}
