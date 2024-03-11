#ifndef DESCRIPTOR_HPATCHESDESCRIPTOREXTRACTOR_HPP
#define DESCRIPTOR_HPATCHESDESCRIPTOREXTRACTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <thread>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include "descriptorProcessor.hpp"
#include "experimentConfig.hpp"
#include "../keypoints/VanillaSIFT.h"
#include "../keypoints/logger.hpp"

/**
 * @class hpatchesDescriptorExtractor
 * @brief Extracts image descriptors using scaled pooling methods.
 *
 * This class provides functionality to extract descriptors from the hpatches dataset
 * the default descriptorExtractor is SIFT, but it can be modified to use other descriptors.
 * in the marked section of the processImage method.
 */
class hpatchesDescriptorExtractor {
public:
    /**
     * Processes a single image to extract descriptors and save them to a file.
     *
     * @param fname The filename (including path) of the image to process.
     * @param seqDirName The name of the sequence directory for organizing output.
     * @param descr_name The descriptorExtractor name used for naming the output directory.
     */
    static void processImage(const std::string& fname, const std::string& seqDirName, const std::string& descr_name,
                             const ExperimentConfig& config);
    /**
     * Processes all images in a given sequence directory.
     *
     * @param seqDirName The name of the directory containing the image sequence to process.
     * @param descr_name The descriptorExtractor name, used for naming the output directory.
     */
    static void processSequenceDirectory(const std::string& seqDirName, const std::string& p, const std::string& descr_name,
                                         const ExperimentConfig& config);
    /**
     * Processes images in multiple sequence directories in parallel.
     *
     * @param descr_name The descriptorExtractor name, used for naming the output directories.
     * @param p The path to the parent directory containing the sequence directories.
     */
    static void processImages(const std::string& descr_name, const std::string& p, const ExperimentConfig& config);
};

#endif //DESCRIPTOR_HPATCHESDESCRIPTOREXTRACTOR_HPP
