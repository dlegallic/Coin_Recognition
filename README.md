# Coin Recognition using Computer Vision
The goal is to recognise coins (euros) using computer vision techniques.<br>
Currently working on coin detection.

## Coin Detection
The coin detection process consist of the following steps :
  1. Edge detection - isolate the edges and output a binary image
  2. Image downscaling - downscaling the image allows for better performances
  3. Hough transform - transform the image in a parameter space to identify shapes

### Edge Detection
Operators are tools to highlight the edges of objects in an image using convolutions.<br>
The most common operators are *Sobel* and *Prewitt* operators, that have been implemented in their vectorized form.<br>
In order to achieve better results we use a multi-stages algorithm called *Canny edge detector* that make the result more robust to noises.
### Hough Transform
For educational purpose both the Line Hough Transform (*LHT*) and Circle Hough Transform (*CHT*) have been implemented.<br>
As their name implies they can detect lines and circle in a given pre-processed image.
