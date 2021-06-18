// #include "itkImage.h"
// #include "itkImageFileReader.h"
// #include "itkRescaleIntensityImageFilter.h"
// #define ENABLE_QUICKVIEW
// #ifdef ENABLE_QUICKVIEW
// #  include "QuickView.h"
// #endif

// #include "vtkAutoInit.h"
// VTK_MODULE_INIT(vtkRenderingOpenGL2);
// VTK_MODULE_INIT(vtkInteractionStyle);

// constexpr unsigned int Dimension = 2;
// using ImageType = itk::Image<unsigned char, Dimension>;

// static void
// CreateImage(ImageType * const image);

// int main(int argc, char * argv[])
// {
//   ImageType::Pointer image;


//   using ReaderType = itk::ImageFileReader<ImageType>;
//   ReaderType::Pointer reader = ReaderType::New();
//   reader->SetFileName("/Users/lijian/Local/ITK/program/data/slices/0/2d_im_t5.mhd");
//   image = reader->GetOutput();


//   using RescaleFilterType = itk::RescaleIntensityImageFilter<ImageType, ImageType>;
//   RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
//   rescaleFilter->SetInput(image);
//   rescaleFilter->SetOutputMinimum(0);
//   rescaleFilter->SetOutputMaximum(255);
//   rescaleFilter->Update();

// #ifdef ENABLE_QUICKVIEW
//   QuickView viewer;
//   viewer.AddImage(reader->GetOutput());
//   // viewer.AddImage(rescaleFilter->GetOutput());
//   viewer.Visualize();
// #endif

//   return EXIT_SUCCESS;
// }

// void
// CreateImage(ImageType * const image)
// {
//   // Create an image with 2 connected components
//   ImageType::IndexType corner = { { 0, 0 } };

//   ImageType::SizeType size;
//   unsigned int        NumRows = 200;
//   unsigned int        NumCols = 300;
//   size[0] = NumRows;
//   size[1] = NumCols;

//   ImageType::RegionType region(corner, size);

//   image->SetRegions(region);
//   image->Allocate();

//   // Make a square
//   for (unsigned int r = 40; r < 100; r++)
//   {
//     for (unsigned int c = 40; c < 100; c++)
//     {
//       ImageType::IndexType pixelIndex;
//       pixelIndex[0] = r;
//       pixelIndex[1] = c;

//       image->SetPixel(pixelIndex, 15);
//     }
//   }
// }

#include "itkLogImageFilter.h"
#include "itkRandomImageSource.h"
#include "itkImageDuplicator.h"
#include "itkMultiThreaderBase.h"
#include "itkImageRegionIterator.h"

constexpr unsigned int Dimension = 2;
using PixelType = unsigned int;
using ImageType = itk::Image<PixelType, Dimension>;

using namespace std;
// calculate log(1+x), where x is pixel value, using LogImageFilter
void
log1xViaLogImageFilter(ImageType::Pointer & image)
{
  // LogImageFilter calculates log(x), so we have to modify the image first
  // by increase its every pixel value by 1, and then apply log filter to it
  itk::ImageRegionIterator<ImageType> it(image, image->GetBufferedRegion());
  for (; !it.IsAtEnd(); ++it)
  {
    it.Set(1 + it.Get());
  }

  // classic filter declaration and invocation
  using LogType = itk::LogImageFilter<ImageType, ImageType>;
  LogType::Pointer logF = LogType::New();
  logF->SetInput(image);
  logF->SetInPlace(true);
  logF->Update();
  image = logF->GetOutput();
  image->DisconnectPipeline();
}

// calculate log(1+x), where x is pixel value, using ParallelizeImageRegion
void
log1xViaParallelizeImageRegion(ImageType::Pointer & image, ImageType::Pointer & image2)
{
  itk::MultiThreaderBase::Pointer mt = itk::MultiThreaderBase::New();
  // ParallelizeImageRegion invokes the provided lambda function in parallel
  // each invocation will contain a piece of the overall region
  mt->ParallelizeImageRegion<Dimension>(
    image->GetBufferedRegion(),
    // here we creat an ad-hoc lambda function to process the region pieces
    // the lambda will have access to variable 'image' from the outer function
    // it will have parameter 'region', which needs to be processed
    [image, image2](const ImageType::RegionType & region) {
      itk::ImageRegionIterator<ImageType> it(image, region);itk::ImageRegionIterator<ImageType> it2(image2, region);
      for (; !it.IsAtEnd(); ++it)
      {
        it.Set(std::log(1 + (it.Get()+it2.Get())/2));
      }
    },
    nullptr); // we don't have a filter whose progress needs to be updated
}

int
main(int, char *[])
{
  int result = EXIT_SUCCESS;

  // create an image
  ImageType::RegionType region = { { 0, 0 }, { 50, 20 } }; // indices zero, size 50x20
  using RandomSourceType = itk::RandomImageSource<ImageType>;
  RandomSourceType::Pointer randomImageSource = RandomSourceType::New();
  randomImageSource->SetSize(region.GetSize());
  // we don't want overflow on 1+x operation, so set max pixel value
  randomImageSource->SetMax(itk::NumericTraits<PixelType>::max() - 1);
  randomImageSource->SetNumberOfWorkUnits(1); // to produce deterministic results
  randomImageSource->Update();

  ImageType::Pointer image = randomImageSource->GetOutput();
  image->DisconnectPipeline();

  // create another image, to be passed to the alternative method
  using DuplicatorType = itk::ImageDuplicator<ImageType>;
  DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(image);
  duplicator->Update();

  ImageType::Pointer clonedImage = duplicator->GetOutput();
  clonedImage->DisconnectPipeline();

    cout<<"start:"<<endl;
    auto begin = std::chrono::high_resolution_clock::now();
  // invoke the two functions
  log1xViaLogImageFilter(image);
      auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin);
    cout<<"runtime:   "<< elapsed.count() * 1e-9 <<endl;

    begin = std::chrono::high_resolution_clock::now();
  log1xViaParallelizeImageRegion(clonedImage, clonedImage);
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin);
    cout<<"runtime:   "<< elapsed.count() * 1e-9 <<endl;
    
  // compare to make sure the results are the same
//   unsigned                                 diffCount = 0;
//   itk::ImageRegionConstIterator<ImageType> it1(image, region);
//   itk::ImageRegionConstIterator<ImageType> it2(clonedImage, region);
//   for (; !it1.IsAtEnd(); ++it1, ++it2)
//   {
//     if (it1.Get() != it2.Get())
//     {
//       std::cerr << "Pixel values are different at index " << it1.GetIndex() << it1.Get() << " vs. " << it2.Get()
//                 << std::endl;
//       //<< "\n\tlog1xViaLogImageFilter's value: " << it1.Get()
//       //<< "\n\tlog1xViaParallelizeImageRegion: " << it2.Get() << std::endl;
//       diffCount++;
//       result = EXIT_FAILURE;
//     }
//   }

//   if (diffCount == 0)
//   {
//     std::cout << "LogImageFilter and ParallelizeImageRegion generate the same result." << std::endl;
//   }
//   else
//   {
//     std::cout << "Discrepancy! " << diffCount << " pixels out of " << region.GetNumberOfPixels() << " are different."
//               << std::endl;
//   }
  return result;
}
