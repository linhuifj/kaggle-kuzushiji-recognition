# kaggle-kuzushiji-recognition

## Method overview
character detection -> get lines -> line recognition -> postprocessing

### Text line detection
codes are in line_det/ directory.

we tried to use retina net and center net to detect characters. The result may be inaccurate, but it doesn't matter. We just merge the boxes with some rules to get vertical lines, and extract each lines out with their position informations stored in file.


## Text recognition
Each extracted line is resized (with padding) to image 32x800. 

## Interesting findings
1. position accuracy
2. data augmentation
3. 
