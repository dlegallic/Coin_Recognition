# Coin Detection using Computer Vision
The goal is to detect coins (euros) using computer vision techniques.<br>

## Coin Detection
The coin detection process consist of the following steps :
  1. Image downscaling - downscaling the image allows for better performances
  2. Edge detection - isolate the edges and output a binary image
  3. Hough transform - transform the image in a parameter space to identify shapes
  4. Local peak identification - finding local peaks in the parameter space
  5. Return identified coins - converting peaks in the parameter space to concrete coins coordinates

### Edge Detection
Operators are tools to highlight the edges of objects in an image using convolutions.<br>
The most common operators are *Sobel* and *Prewitt* operators, that have been implemented in their vectorized form.<br>
In order to achieve better results we use a multi-stages algorithm called *Canny edge detector* that make the result more robust to noises.

### Hough Transform
For educational purpose both the Line Hough Transform (*LHT*) and Circle Hough Transform (*CHT*) have been implemented.<br>
As their name implies they can detect lines and circle in a given pre-processed image.

### Local Peak Identification
We use a greedy Non Maximum Suppression algorithm. In the 3d Hough parameter space, we find peaks and flatten their neighbourhood until there are no more points over a given threshold.<br>
This algorithm is neither the most accurate nor the fastest, but is easy to implement and is sufficient for the given task.

## Evaluation of our process
In order to evaluate the performances of our process, we use a labeled test dataset, on which we train, and a labeled validation dataset, reserved for the final evaluation.<br>
The implemented metric is a simple Intersection over Union (IoU) that will tell us how accurately our models predicts coins.
Other metrics, like the distance between predicted and actual number of coins could be implemented in the future.
Furthermore, we use a simple openCV process similar to our own to compare their speed and accuracy.

# How to Use
## Installation
Install dependencies using:

```terminal
pip install -r requirements.txt
```
## Running the program
From the scripts folder, run:
```python3
python3 main.py --full
```
This will execute the program with default settings:
- Slow `manual` Python implementation of Hough Transform
- Runs on the `test` dataset

  
## Options
### Change implementation
To use OpenCV implementation, simply run :
```python3
python3 main.py --full -algo 'opencv'
```

### Change dataset
To run on the validation dataset, run :
```python3
python3 main.py --full -dataset 'validation'
```
### See detected coins
If you want to see each image's detected coins you can run :
```python3
python3 main.py --full -graphic
```
### Run on a single file
If you only wish to detect coins on a single image, run:
```python3
python3 main.py --file $file_path$
```
Note that in this mode, the resulting image will obviously always be shown. 

### Notes
For more details, and option settings, run :
```python3
python3 main.py --help
```
