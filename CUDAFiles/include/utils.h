#ifndef UTILS_H
#define UTILS_H

#include <vector>

void D2F(unsigned int count, double* input, float* output);
void U82D(unsigned int count, unsigned char* in, double* out);
std::vector<double> loadImage(const char* filename, unsigned int width, unsigned int height);

#endif