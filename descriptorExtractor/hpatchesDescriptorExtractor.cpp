#include "hpatchesDescriptorExtractor.hpp"

 float PATCH_SIZE = 65.0f; // original h patches patch size 65 color 66 don't know why they did not use even number

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
    for (int r = 0; r < im.rows; r += int(PATCH_SIZE)) {
        for (int c = 0; c < im.cols; c += int(PATCH_SIZE)) {
            cv::Point2f center(c + PATCH_SIZE/2, r + PATCH_SIZE/2);
            keypoints.emplace_back(center, PATCH_SIZE);
        }
    }

//    // Visual check of the keypoints
//    cv::Mat imageWithKeypoints;
//    drawKeypoints(im, keypoints, imageWithKeypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//    imshow("Keypoints", imageWithKeypoints);
//    waitKey(0);



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

    // DONE: Try vanilla SIFT here and test -> FAILED
//    //#############################################################
//    if (im.channels() == 3) {
//        // Image is in BGR format, convert it to grayscale
//        cv::cvtColor(im, im, cv::COLOR_BGR2GRAY);
//    } else if (im.channels() == 4) {
//        // For example, convert BGRA to grayscale
//        cv::cvtColor(im, im, cv::COLOR_BGRA2GRAY);
//    }
//
//    // init
//    Ptr<VanillaSIFT> vanillaSiftExtractor = VanillaSIFT::create();
//
//    cv::Mat descriptors;
//
//    vanillaSiftExtractor->compute(im, keypoints, descriptors);
//    //############################################################


    // Create the descriptor extractor based on the config
    auto descriptorExtractor = config.createDescriptorExtractor();

    // Process descriptors
    cv::Mat descriptors = DescriptorProcessor::processDescriptors(im, keypoints, descriptorExtractor, config.descriptorOptions);

    // No modifications needed beyond this point
    // ########################################################

     // filter fo Nan values and set them to 0
     int nanCount = 0;
     for (int i = 0; i < descriptors.rows; ++i) {
         for (int j = 0; j < descriptors.cols; ++j) {
             if (std::isnan(descriptors.at<float>(i, j))) {
                 descriptors.at<float>(i, j) = 0;
                    nanCount++;
             }
         }
     }

    if (nanCount > 0) {
        std::cout << "Found " << nanCount << " NaN values in the descriptors.\n";
    }

    // Check for all zero descriptors
    int zeroCount = 0;
    for (int i = 0; i < descriptors.rows; ++i) {
        bool allZero = true;
        for (int j = 0; j < descriptors.cols; ++j) {
            if (descriptors.at<float>(i, j) != 0) {
                allZero = false;
                break;
            }
        }
        if (allZero) {
            zeroCount++;
            // print fname of the image
            std::cout << "All zero descriptor found in " << fname << std::endl;
        }
    }

    if (zeroCount > 0) {
        std::cout << "Found " << zeroCount << " all-zero descriptors.\n";
    }

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

