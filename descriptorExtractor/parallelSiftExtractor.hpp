#ifndef PARALLEL_SIFT_EXTRACTOR_HPP
#define PARALLEL_SIFT_EXTRACTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp> // Include SIFT header
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

class ParallelSiftExtractor {
public:
    static void processImage(const std::string& fname, const std::string& seqDirName, const std::string& descr_name) {
        std::cout << "Extracting descriptors for " << fname << std::endl;
        cv::Mat im = cv::imread(fname, 0);

        if (im.empty()) {
            std::cerr << "Error: Unable to read image " << fname << std::endl;
            return;
        }

        // Extracting filename and type part
        std::vector<std::string> strs;
        boost::split(strs, fname, boost::is_any_of("/"));
        std::string img_name = strs.back();
        std::vector<std::string> strs_;
        boost::split(strs_, img_name, boost::is_any_of("."));
        std::string tp = strs_[0];

        std::string outputDirectory = "../results/" + descr_name + "/" + seqDirName;
        boost::filesystem::create_directories(outputDirectory);
        std::string outputFile = outputDirectory + "/" + tp + ".csv";

        std::ofstream f(outputFile);
        if (!f.is_open()) {
            std::cerr << "Error: Unable to open file for writing " << outputFile << std::endl;
            return;
        }

        auto sift = cv::SIFT::create();
        std::vector<cv::KeyPoint> keypoints;
        std::stringstream ss; // Stringstream to accumulate descriptorExtractor data

        // Collect keypoints
        for (int r = 0; r < im.rows; r += 65) {
            for (int c = 0; c < im.cols; c += 65) {
                cv::Point2f center(c + 32.5f, r + 32.5f);
                keypoints.emplace_back(center, 65.0f);
            }
        }

        // Compute descriptors for all keypoints
        cv::Mat descriptors;
        sift->compute(im, keypoints, descriptors);

        // l1 normalization of descriptors
        cv::normalize(descriptors, descriptors, 1, 0, cv::NORM_L1);

        cv::Mat descriptors_rooted = cv::Mat::zeros(descriptors.rows, descriptors.cols, CV_32FC1);

        // square root normalization of descriptors
        for (int i = 0; i < descriptors.rows; ++i) {
            for (int j = 0; j < descriptors.cols; ++j) {
                descriptors_rooted.at<float>(i, j) = sqrt(descriptors.at<float>(i, j));
            }
        }


        // Accumulate descriptorExtractor data into stringstream
        for (int i = 0; i < descriptors_rooted.rows; ++i) {
            for (int j = 0; j < descriptors_rooted.cols; ++j) {
                ss << descriptors_rooted.at<float>(i, j);
                if (j < descriptors_rooted.cols - 1) ss << ",";
            }
            ss << "\n";
        }

        // Write the accumulated data to the file
        f << ss.str();
        f.close();
    }


    static void processSequenceDirectory(const std::string& seqDirName, const std::string& descr_name) {
        std::string fullPath = "../data/" + seqDirName; // Assuming data is in ../data/
        for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(fullPath), {})) {
            std::string fname = entry.path().string();
            std::string ext = entry.path().extension().string();
            if (ext == ".png") {
                processImage(fname, seqDirName, descr_name);
            }
        }
    }

    static void processImages(const std::string& descr_name, const std::string& p) {
        std::cout << "Processing directory: " << p << "\n";

        std::vector<std::string> seqDirectories;
        if (boost::filesystem::is_directory(p)) {
            for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {})) {
                if (boost::filesystem::is_directory(entry.path())) {
                    std::vector<std::string> pathParts;
                    boost::split(pathParts, entry.path().string(), boost::is_any_of("/"));
                    std::string seqDirName = pathParts.back();
                    seqDirectories.push_back(seqDirName);
                    std::cout << "Found sequence directory: " << seqDirName << "\n";
                }
            }
        } else {
            std::cout << "Provided path is not a directory.\n";
            return; // Exit if path is not a directory
        }

        std::vector<std::thread> threads;
        for (const auto& seqDir : seqDirectories) {
            threads.emplace_back(processSequenceDirectory, seqDir, descr_name);
        }

        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        std::cout << "Parallel processing completed.\n";
    }
};

#endif // PARALLEL_SIFT_EXTRACTOR_HPP
