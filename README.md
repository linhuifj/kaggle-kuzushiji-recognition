# kaggle-kuzushiji-recognition

## Method overview
character detection -> get lines -> line recognition -> postprocessing

### Text line detection
We first detect bounding boxes of each chacacter, then we group the boxes to merge vertical lines. These lines will be extracted and recognized. 

details are in line_det/ directory.

## Text recognition
Each extracted line is resized (with padding) to image 32x800, and fed into CRNN(CNN + LSTM + CTC) model for recognition. Attention is added as multi-task learning for improving the alignment accuracy of CTC model.


## Decode
We use beam search + language model for decoding. Language model is trained by kenlm with the vertical line texts from training data.


## Post Processing
usage: adjust_center/adjust.py output.csv > outout_new.csv
We binarize the images to get the stroke of the characters. Then for each coordinate of the resuling character, we adjust it's position to the nearest stroke. This pocessing will increase the score by abount 0.01.



## Import Points
1. Position accuracy

The CRNN model is used for recognition. It takes input with 32x800 and outputs 200x4788. The model without lstm layer has accuate position output but lower accuracy. When lstm is added, the position drifts and become inaccurate. Adding attention output as a multitask leaning objective will increase the position accuracy of CRNN.

2. Data augmentation

Random brightness, contrast, distortion, scaling are added to augment the training data. 

3. Regularization

Dropout, cutout, weight decay and early stopping are added to prevent overfitting.

4. Network architecture

Attention model performs worse than ctc. Resnet is better than VGG, but resnet-xt and squeeze-excitation network does not improve performance. LSTM layes more than 2 will make the position inaccurate.



