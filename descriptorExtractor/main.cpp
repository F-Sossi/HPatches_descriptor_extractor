#include <vector>
#include <opencv2/core.hpp>
#include <string>
#include "experimentConfig.hpp"
#include "hpatchesDescriptorExtractor.hpp"

// TODO - NEED to rerun the tests on the color_data set found we had hardcoded to the data folder in the hpatchesDescriptorExtractor.cpp

// Helper function to convert enum and settings to string for naming
std::string poolingStrategyToString(PoolingStrategy strategy) {
    switch (strategy) {
        case NONE: return "None";
        case AVERAGE_POOLING: return "Avg";
        case MAX_POOLING: return "Max";
        case DOMAIN_SIZE_POOLING: return "Dom";
        case STACKING: return "Stack";
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
        case R_NONE: return "NoRoot";
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

std::string descriptorTypeToString(DescriptorType type) {
    switch (type) {
        case DESCRIPTOR_SIFT: return "SIFT";
        case DESCRIPTOR_ORB: return "ORB";
        case DESCRIPTOR_SURF: return "SURF";
        case DESCRIPTOR_vSIFT: return "vSIFT";
            // Add more cases as needed
        default: return "Unknown";
    }
}

std::string imageTypeToString(ImageType imageType) {
    switch (imageType) {
        case COLOR: return "CLR";
        case BW: return "BW";
        default: return "Unknown";
    }
}


int main() {
    // Define all possible options
    std::vector<PoolingStrategy> poolingStrategies = {NONE};
    std::vector<NormalizationStage> normalizationStages = {NO_NORMALIZATION};
    std::vector<RootingStage> rootingStages = {R_NONE}; // R_AFTER_POOLING was worse in all cases
    std::vector<int> normTypes = {cv::NORM_L1}; // L@ Norm was worse in all cases
    std::vector<DescriptorType> descriptorTypes = {DESCRIPTOR_vSIFT}; // Example descriptor types

    // Data is the original data set and color_data is the same data set in color
    std::string directoryPath = "../data2";

    // Iterate over all combinations of options, including descriptor types
    for (auto &descriptorType: descriptorTypes) {
        for (auto &pooling: poolingStrategies) {
            for (auto &normalization: normalizationStages) {
                for (auto &rooting: rootingStages) {
                    for (auto &normType: normTypes) {
                        // Configure the descriptor options
                        DescriptorOptions options;
                        options.poolingStrategy = pooling;
                        options.scales = {0.5f, 1.0f, 1.5f}; // Example scales, modify as needed
                        options.normType = normType;
                        options.normalizationStage = normalization;
                        options.rootingStage = rooting;
                        // Set the color type to match the descriptor type BW or COLOR ex: SIFT is BW
                        options.imageType = BW;
                        options.descriptorColorSpace = D_BW;

                        // Create experiment configuration with descriptor options and type
                        ExperimentConfig config(options, descriptorType);

                        // Create a descriptive experiment name
                        std::string descriptorName = descriptorTypeToString(descriptorType) + "-" +
                                imageTypeToString(options.imageType) + "-" +
                                poolingStrategyToString(pooling) + "-" +
                                normalizationStageToString(normalization) + "-" +
                                rootingStageToString(rooting) + "-" +
                                normTypeToString(normType) + "-Baseline";

                        // Run the descriptor extraction process
                        hpatchesDescriptorExtractor::processImages(descriptorName, directoryPath, config);
                    }
                }
            }
        }
    }

    return 0;
}