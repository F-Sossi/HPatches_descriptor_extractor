#include "logger.hpp"

bool Logger::isEnabled = false; // Definition of the static member variable
bool Logger::isErrorEnabled = true; // Definition of the static member variable

void Logger::Log(const std::string& message) {
    if (!isEnabled) return;

    std::ofstream logFile("vanilla_sift_log.txt", std::ios::out | std::ios::app);
    if (logFile.is_open()) {
        logFile << message << std::endl;
        logFile.close();
    } else {
        std::cerr << "Unable to open log file." << std::endl;
    }
}

void Logger::LogErr(const std::string& message) {
    if (!isErrorEnabled) return;

    std::ofstream logFile("vanilla_error_log.txt", std::ios::out | std::ios::app);
    if (logFile.is_open()) {
        logFile << message << std::endl;
        logFile.close();
    } else {
        std::cerr << "Unable to open log file." << std::endl;
    }
}

