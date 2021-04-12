#include "settings.h"
#include "json.hpp"
#include <fstream>

Settings::Settings(const char* filename){
    j = loadJson(filename);
}

json Settings::loadJson(const char* filename){
    std::ifstream in(filename);
    json j;
    in >> j;
    return(j);
}
 
std::string Settings::parseString(const char* param){
    return j.at(param).get<std::string>();
}

std::vector<double> Settings::parseVecDouble(const char* param){
    return j.at(param).get<std::vector<double>>();
}

double Settings::parseDouble(const char* param){
    return j.at(param).get<double>();
}

int Settings::parseInt(const char* param){
    return j.at(param).get<int>();
}