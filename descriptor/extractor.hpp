//
// Created by frank on 1/13/24.
//

#ifndef DESCRIPTOR_EXTRACTOR_HPP
#define DESCRIPTOR_EXTRACTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp> // Include SIFT header
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

class Extractor {
public:
    static void processImages(const std::string& descr_name, const std::string& p) {
        std::cout << "Processing directory: " << p << "\n";

        std::vector<std::string> seqs;
        if (boost::filesystem::is_directory(p)) {
            for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {})) {
                if (boost::filesystem::is_directory(entry.path())) {
                    seqs.push_back(entry.path().string());
                    std::cout << "Found sequence directory: " << entry.path().string() << "\n";
                }
            }
        } else {
            std::cout << "Provided path is not a directory.\n";
            return; // Exit if path is not a directory
        }

        std::cout << "Found " << seqs.size() << " sequences.\n";

        for (auto const& sq : seqs) {
            std::vector<std::string> seq_splits;
            boost::split(seq_splits, sq, boost::is_any_of("/"));
            std::string seq_name = seq_splits.back(); // Extract seq_name from sq

            for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(sq), {})) {
                std::string fname = entry.path().string();
                std::string ext = entry.path().extension().string(); // Use path::extension()
                if (ext == ".png") {
                    std::cout << "Extracting descriptors for " << fname << std::endl;
                    cv::Mat im = cv::imread(fname, 0);

                    if (im.empty()) {
                        std::cerr << "Error: Unable to read image " << fname << std::endl;
                        continue; // Skip this file if it can't be read
                    }

                    std::vector<std::string> strs;
                    boost::split(strs, fname, boost::is_any_of("/"));
                    std::string img_name = strs.back();
                    std::vector<std::string> strs_;
                    boost::split(strs_, img_name, boost::is_any_of("."));
                    std::string tp = strs_[0];

                    std::ofstream f;
                    std::string outputDirectory = "../results/";
                    outputDirectory += descr_name;
                    outputDirectory += "/";
                    outputDirectory += seq_name; // Concatenate using +=

                    boost::filesystem::create_directories(outputDirectory); // Ensure the directory exists

                    std::string outputFile = outputDirectory + "/" + tp + ".csv";
                    f.open(outputFile);
                    if (!f.is_open()) {
                        std::cerr << "Error: Unable to open file for writing " << outputFile << std::endl;
                        continue; // Skip this file if it can't be opened for writing
                    }

//                    for (int r = 0; r < im.rows; r += 65) {
//                        cv::Mat patch = im(cv::Range(r, r + 65), cv::Range(0, 65));
//                        cv::Scalar mi, sigma;
//                        cv::meanStdDev(patch, mi, sigma);
//                        f << mi[0] << "," << sigma[0] << std::endl;
//                    }

                    // Create a SIFT object outside the loop
                    auto sift = cv::SIFT::create();

                    for (int r = 0; r < im.rows; r += 65) {
                        if (r + 65 > im.rows) break; // Make sure the patch does not exceed image boundaries
                        for (int c = 0; c < im.cols; c += 65) {
                            if (c + 65 > im.cols) break; // Same check for column

                            // Define the center and size of the patch
                            cv::Point2f center(c + 32.5f, r + 32.5f); // Center of the patch
                            std::vector<cv::KeyPoint> keypoints;
                            keypoints.emplace_back(center, 65.0f); // Size of the patch as keypoint size

                            // Compute SIFT descriptor for this keypoint
                            cv::Mat descriptors;
                            sift->compute(im, keypoints, descriptors);

                            // Write the descriptor to the file
                            for (int i = 0; i < descriptors.rows; ++i) {
                                for (int j = 0; j < descriptors.cols; ++j) {
                                    f << descriptors.at<float>(i, j);
                                    if (j < descriptors.cols - 1) f << ",";
                                }
                                f << std::endl;
                            }
                        }
                    }

                    f.close();
                }
            }
        }

        std::cout << "Processing completed.\n";
    }
};

#endif //DESCRIPTOR_EXTRACTOR_HPP
