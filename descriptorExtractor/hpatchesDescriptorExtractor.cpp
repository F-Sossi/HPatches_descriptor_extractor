#include "hpatchesDescriptorExtractor.hpp"

 void hpatchesDescriptorExtractor::processImage(const std::string& fname, const std::string& seqDirName,
                                                const std::string& descr_name, const ExperimentConfig& config){
    std::cout << "Extracting descriptors for " << fname << std::endl;
    cv::Mat im = cv::imread(fname, 0);

    if (im.empty()) {
        std::cerr << "Error: Unable to read image " << fname << std::endl;
        return;
    }

    // TODO - remove this check
    if (im.depth() != CV_8U) {
        std::cerr << "Error: Image depth is not CV_8U for image " << fname << std::endl;
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

    std::vector<cv::KeyPoint> keypoints;

    // Collect keypoints centered on each patch
    for (int r = 0; r < im.rows; r += 65) {
        for (int c = 0; c < im.cols; c += 65) {
            cv::Point2f center(c + 32.5f, r + 32.5f);
            keypoints.emplace_back(center, 65.0f);
        }
    }

//     // Collect keypoints centered on each patch
//     for (int r = 0; r < im.rows; r += 65) {
//         for (int c = 0; c < im.cols; c += 65) {
//             cv::Point2f center(c + 32.5f, r + 32.5f);
//             float diameter = 30.0f * 2; // Set the diameter to be slightly less than the block size
//             keypoints.emplace_back(center, diameter);
//         }
//     }

    //########################################################
    // This section modified to use DescriptorProcessor
    // With the descriptorExtractor and options you would like to test
    //########################################################

    // Create the descriptor extractor based on the config
    auto descriptorExtractor = config.createDescriptorExtractor();

    // Process descriptors
    cv::Mat descriptors = DescriptorProcessor::processDescriptors(im, keypoints, descriptorExtractor, config.descriptorOptions);

    // No modifications needed beyond this point
    // ########################################################

    // Accumulate descriptorExtractor data into string stream
    std::stringstream ss;
    for (int i = 0; i < descriptors.rows; ++i) {
        for (int j = 0; j < descriptors.cols; ++j) {
            ss << descriptors.at<float>(i, j);
            if (j < descriptors.cols - 1) ss << ",";
        }
        ss << "\n";
    }

    // Write the accumulated data to the file
    f << ss.str();
    f.close();
}

void hpatchesDescriptorExtractor::processSequenceDirectory(const std::string& seqDirName, const std::string& p, const std::string& descr_name,
                                                           const ExperimentConfig& config) {
    std::string fullPath = p + "/" + seqDirName;

    std::string outputDirectory = "../results/" + descr_name + "/" + seqDirName;
    try {
        boost::filesystem::create_directories(outputDirectory);
        //std::cout << "Created directory: " << outputDirectory << "\n";
    } catch (const boost::filesystem::filesystem_error& e) {
        std::cerr << "Failed to create directory '" << outputDirectory << "': " << e.what() << std::endl;
    }

    for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(fullPath), {})) {
        std::string fname = entry.path().string();
        std::string ext = entry.path().extension().string();
        if (ext == ".png") {
            processImage(fname, seqDirName, descr_name, config);
        }
    }
}

void hpatchesDescriptorExtractor::processImages(const std::string& descr_name, const std::string& p, const ExperimentConfig& config) {
    std::cout << "Processing directory: " << p << "\n";

    // Create the top-level directory for the descriptor name
    std::string descrDirectory = "../results/" + descr_name;
    try {
        boost::filesystem::create_directories(descrDirectory);
    } catch (const boost::filesystem::filesystem_error& e) {
        std::cerr << "Failed to create directory '" << descrDirectory << "': " << e.what() << std::endl;
        return; // Exit on failure
    }

    std::vector<std::string> seqDirectories;
    if (boost::filesystem::is_directory(p)) {
        for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {})) {
            if (boost::filesystem::is_directory(entry.path())) {
                // Extract the directory name correctly
                std::string fullPath = entry.path().string();
                std::vector<std::string> pathParts;
                boost::split(pathParts, fullPath, boost::is_any_of("/\\"));
                std::string seqDirName = pathParts.back();
                // Handle case where path ends with a slash and last element is empty
                if (seqDirName.empty() && pathParts.size() > 1) {
                    seqDirName = pathParts[pathParts.size() - 2];
                }
                seqDirectories.push_back(seqDirName);
                std::cout << "Found sequence directory: " << seqDirName << "\n";
            }
        }
    } else {
        std::cout << "Provided path is not a directory.\n";
        return; // Exit if path is not a directory
    }

    if (config.useMultiThreading) {
        // Multithreading processing
        std::vector<std::thread> threads;
        unsigned int maxThreads = std::thread::hardware_concurrency();

        for (const auto& seqDirName : seqDirectories) {
            if (threads.size() >= maxThreads) {
                for (auto& thread : threads) {
                    thread.join();
                }
                threads.clear();
            }

            threads.emplace_back([&] {
                processSequenceDirectory(seqDirName, p, descr_name, config);
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        std::cout << "Parallel processing completed.\n";
    } else {
        // Single-threaded processing
        for (const auto& seqDirName : seqDirectories) {
            processSequenceDirectory(seqDirName, p, descr_name, config);
        }

        std::cout << "Single-threaded processing completed.\n";
    }

    std::cout << "Single-threaded processing completed.\n";
}

// TODO erase old code

//void hpatchesDescriptorExtractor::processImages(const std::string& descr_name, const std::string& p, const ExperimentConfig& config) {
//    std::cout << "Processing directory: " << p << "\n";
//
//    // Create the top-level directory for the descriptor name
//    std::string descrDirectory = "../results/" + descr_name;
//    try {
//        boost::filesystem::create_directories(descrDirectory);
//    } catch (const boost::filesystem::filesystem_error& e) {
//        std::cerr << "Failed to create directory '" << descrDirectory << "': " << e.what() << std::endl;
//        return; // Exit on failure
//    }
//
//    std::vector<std::string> seqDirectories;
//    if (boost::filesystem::is_directory(p)) {
//        for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {})) {
//            if (boost::filesystem::is_directory(entry.path())) {
//                // Extract the directory name correctly
//                std::string fullPath = entry.path().string();
//                std::vector<std::string> pathParts;
//                boost::split(pathParts, fullPath, boost::is_any_of("/\\"));
//                std::string seqDirName = pathParts.back();
//                // Handle case where path ends with a slash and last element is empty
//                if (seqDirName.empty() && pathParts.size() > 1) {
//                    seqDirName = pathParts[pathParts.size() - 2];
//                }
//                seqDirectories.push_back(seqDirName);
//                std::cout << "Found sequence directory: " << seqDirName << "\n";
//            }
//        }
//    } else {
//        std::cout << "Provided path is not a directory.\n";
//        return; // Exit if path is not a directory
//    }
//
//    unsigned int maxThreads = std::thread::hardware_concurrency();
//    std::vector<std::thread> threads;
//
//    for (const auto& seqDirName : seqDirectories) {
//        if (threads.size() >= maxThreads) {
//            for (auto& thread : threads) {
//                thread.join();
//            }
//            threads.clear();
//        }
//
//        // Use the sequence directory name directly for threading
//        threads.emplace_back([&, seqDirName] {
//            processSequenceDirectory(seqDirName, descr_name, config);
//        });
//    }
//
//    // Ensure all threads are completed
//    for (auto& thread : threads) {
//        thread.join();
//    }
//
//    std::cout << "Parallel processing completed.\n";
//}

