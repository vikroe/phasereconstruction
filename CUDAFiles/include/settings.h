#ifndef SETTINGS_H
#define SETTINGS_H

#include "json.hpp"
#include <vector>
#include <string.h>

using json = nlohmann::json;

class Settings{
    private:
        json j;
        json loadJson(const char* filename);
    public:
        Settings(const char* filename);

        // TODO: make this not suck
        std::string parseString(const char* param);
        std::vector<double> parseVecDouble(const char* param);
        int parseInt(const char* param);
        double parseDouble(const char* param);
};

#endif