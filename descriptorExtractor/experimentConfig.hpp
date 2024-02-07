#ifndef EXPERIMENT_CONFIG_HPP
#define EXPERIMENT_CONFIG_HPP

#include "descriptorProcessor.hpp" // Make sure this includes the DescriptorOptions definition

/**
 * @struct ExperimentConfig
 * @brief Encapsulates configuration options for descriptor processing experiments.
 *
 * This structure holds all necessary configuration options that affect
 * descriptor processing, allowing for easy modification of experiments.
 */
struct ExperimentConfig {
    DescriptorOptions descriptorOptions;

    // Constructor
    ExperimentConfig() = default;

    // Initialize with specific descriptor options
    ExperimentConfig(const DescriptorOptions& options) : descriptorOptions(options) {}

    // Helper method to set descriptor options directly
    void setDescriptorOptions(const DescriptorOptions& options) {
        descriptorOptions = options;
    }

    // Add methods to configure specific options if required
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
};

#endif // EXPERIMENT_CONFIG_HPP

