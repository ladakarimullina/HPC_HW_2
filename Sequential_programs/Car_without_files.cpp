
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <thread>

#include <stdio.h>
#include <omp.h>

#define RGB_COMPONENT_COLOR 255

static const auto THREADS = std::thread::hardware_concurrency();

struct PPMPixel {
  int red;
  int green;
  int blue;
};

typedef struct {
  int x, y, all;
  PPMPixel *data;
} PPMImage;

void readPPM(const char *filename, PPMImage &img){
    std::ifstream file(filename);
    if (file) {
        std::string s;
        int rgb_comp_color;
        file >> s;
        if (s != "P3") {
            std::cout << "error in format" << std::endl;
            exit(9);
        }
        file >> img.x >> img.y;
        file >> rgb_comp_color;
        img.all = img.x * img.y;
        std::cout << s << std::endl;
        std::cout << "x=" << img.x << " y=" << img.y << " all=" << img.all
                  << std::endl;
        img.data = new PPMPixel[img.all];
        for (int i = 0; i < img.all; i++) {
            file >> img.data[i].red >> img.data[i].green >> img.data[i].blue;
        }
    } else {
        std::cout << "the file:" << filename << "was not found" << std::endl;
    }
    file.close();
}



void animatePPM(PPMImage &img){
    // Temporary storage for the last column of pixels
    PPMPixel *temp = new PPMPixel[img.y];
    for (int i = 0; i < img.y; ++i) {
        // Save last column
        temp[i] = img.data[i * img.x + (img.x - 1)];
    }
    #pragma omp parallel for
    // Shifting columns to the right
    for (int i = 0; i < img.y; ++i) {
        for (int j = img.x - 1; j > 0; --j) {
            img.data[i * img.x + j] = img.data[i * img.x + (j - 1)];
        }
    }
    #pragma omp parallel for
    // The last column goes the first column position
    for (int i = 0; i < img.y; ++i) {
        img.data[i * img.x] = temp[i];
    }

    delete[] temp;
}

void writePPM(const char *filename, PPMImage &img) {
    std::ofstream file(filename, std::ofstream::out);
    file << "P3" << std::endl;
    file << img.x << " " << img.y << " " << std::endl;
    file << RGB_COMPONENT_COLOR << std::endl;
    for (int i = 0; i < img.all; i++) {
        file << img.data[i].red << " " << img.data[i].green << " "
             << img.data[i].blue << (((i + 1) % img.x == 0) ? "\n" : " ");
    }
    file.close();
}

int main(int argc, char *argv[]) {
    double start, stop;
    start = omp_get_wtime();
    PPMImage image;
    readPPM("car.ppm", image);
    int nframes = 30;
    for (int frame = 0; frame<nframes; ++frame){
      animatePPM(image);
      std::string filename = "new_car_" + std::to_string(frame) + ".ppm";
    }
    //writePPM("new_car.ppm", image);
    //convert -delay 20 -loop 0 new_car_*.ppm animation.gif
    //system("convert -delay 20 -loop 0 new_car_*.ppm car_animation.gif");
    stop = omp_get_wtime();
    printf("total time without command writePPM(): %f\n ", stop - start);
    return 0;
}
