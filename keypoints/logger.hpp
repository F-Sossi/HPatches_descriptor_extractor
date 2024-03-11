#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <fstream>
#include <string>

class Logger {
public:
    static bool isEnabled; // Declaration of the static member
    static bool isErrorEnabled; // Declaration of the static member

    static void Log(const std::string& message);
    static void LogErr(const std::string& message);
};


#endif // LOGGER_HPP

