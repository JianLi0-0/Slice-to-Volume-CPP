#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include "optim.hpp"

#include <iostream>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "ITKImgProcess.h"
#include "Optimization.h"

double ackley_fn(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* opt_data)
{
    const double x = vals_inp(0);
    const double y = vals_inp(1);
    const double pi = 3.1415926;
    double obj_val = -20*std::exp( -0.2*std::sqrt(0.5*(x*x + y*y)) ) - std::exp( 0.5*(std::cos(2*pi*x) + std::cos(2*pi*y)) ) + 22.718282L;
    return obj_val;
}


int main(int, char * argv[])
{
    // using PixelType = unsigned char;
    // constexpr unsigned int Dimension = 3;
    // using ImageType = itk::Image<PixelType, Dimension>;
    using ReaderType = itk::ImageFileReader<VolumeType>;
    ReaderType::Pointer reader = ReaderType::New();
    // const char * filename = argv[1];
    reader->SetFileName("/Users/lijian/Local/ITK/program/data/volumen/thyroid.mhd");
    reader->Update();
    
    VolumeType::Pointer volume = reader->GetOutput();
    const typename VolumeType::SpacingType spacing = volume->GetSpacing();
    TransformType::Pointer transformation = TransformType::New();
    transformation->SetIdentity();
    transformation->SetRotation(0.11,0.12,0.13);
    itk::Vector<double, 3> t;
    t[0] = 5;t[1] = 5;t[2] = 40;
    transformation->SetTranslation(t);
    
    // std::cout << transformation << std::endl;
    float sliceWidth = 256;
    float sliceHeight = 256;
    SliceType::SpacingType outputSpacing;
    // outputSpacing[0] = 
    // try
    // {
        // std::cout << volume << std::endl;


    Optimization optimization;

    cout<<"start:"<<endl;
    auto begin = std::chrono::high_resolution_clock::now();
    auto V1 = ExtractSliceFromVolume(volume, transformation, sliceWidth, sliceHeight, spacing);
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin);
    cout<<"runtime:   "<< elapsed.count() * 1e-9 <<endl;

    t[2] = 60;
    transformation->SetTranslation(t);
    auto V2 = ExtractSliceFromVolume(volume, transformation, sliceWidth, sliceHeight, spacing);

    begin = std::chrono::high_resolution_clock::now();
    auto sum = SumOfSquaredDifferences(V1, V2, sliceWidth, sliceHeight);
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin);
    cout<<"SumOfSquaredDifferences runtime:   "<< elapsed.count() * 1e-9 <<endl;
    cout << "sum: " << sum << endl;

    begin = std::chrono::high_resolution_clock::now();
    sum = SSD2(V1, V2, sliceWidth, sliceHeight, optimization.squaredDiffImg);
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin);
    cout<<"SSD2 runtime:   "<< elapsed.count() * 1e-9 <<endl;
    cout << "sum: " << sum << endl;

    begin = std::chrono::high_resolution_clock::now();
    sum = SSD3(V1, V2);
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin);
    cout<<"SSD3 runtime:   "<< elapsed.count() * 1e-9 <<endl;
    cout << "sum: " << sum << endl;
    

    
    // using WriterType = itk::ImageFileWriter<VolumeType>;
    // typename WriterType::Pointer writer = WriterType::New();
    // writer->SetFileName("/Users/lijian/Local/ITK/program/data/volumen/new_thyroid.mhd");
    // writer->SetInput(tempV);
    // try
    // {
    //     writer->Update();
    // }
    // catch (itk::ExceptionObject & error)
    // {
    //     std::cerr << "Error: " << error << std::endl;
    //     return EXIT_FAILURE;
    // }


    // Optimization optimization;
    VolumeType::Pointer goalSlice = V1;
    Eigen::VectorXd initialTransform;
    optimization.SetVolume(volume);
    cout << "Ground Truth: \n " << optimization.ITKTransformToEigen(transformation) << endl;
    t[0] = 7; t[1] = 7; t[2] = 43;  transformation->SetRotation(0.107,0.117,0.117);
    transformation->SetTranslation(t);
    initialTransform =  optimization.ITKTransformToEigen(transformation);
    cout << "initialTransform: \n " << initialTransform << endl;
    
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    auto success = optimization.Optimize(goalSlice, initialTransform);
    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    
    if (success) {
        std::cout << "de: Registration successfully.\n"
                  << "elapsed time: " << elapsed_seconds.count() << "s\n";
    } else {
        std::cout << "de: Ackley test completed unsuccessfully." << std::endl;
    }
    std::cout << "\nde: solution :\n" << initialTransform << std::endl;


    return 0;
}
