### Method

1. A detector is trained to find all words in the input image. Here [HTC(Hybrid Task Cascade)](http://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Hybrid_Task_Cascade_for_Instance_Segmentation_CVPR_2019_paper.html) is adopted, which is the SOTA two-stage object detection method;
2. Words are grouped into different lines, using the text line constructing algorithm used by [CTPN](https://arxiv.org/abs/1609.03605);
3. Line heights and "linetop" line are calculated through simple regression, based on all word bounding boxes of each text line. Then each line can have a "bounding box", as shown below. Since text lines may not be align with the y-axis of image, the bounding box may be slant.
!["Bounding boxes" of lines](https://note.youdao.com/yws/public/resource/99266270026a0c1b8de576f7ef204bc0/xmlnote/9C0176F0D7424D0D957002CE10A5D934/15013)
4. Lines are cropped from input image and line-wise recognition is done. When cropping the lines out, each pixel coord of "linetop" line(the red line in the image below) are recorded for mapping the predicted positions of each word in the cropped line back to the whole page image, and the cropped line images are all non-slant as shown below.  
![linetop](https://note.youdao.com/yws/public/resource/99266270026a0c1b8de576f7ef204bc0/xmlnote/8DBC512B6FF14866BE79AFD117601DAD/15030)
![cropped img](https://note.youdao.com/yws/public/resource/99266270026a0c1b8de576f7ef204bc0/xmlnote/091B54C953F74B009D9812449155E7C6/15032)
5. It is noticed that some lines have overlaps of large area, which may lead to duplicate word recognition on same position. To alleviate this, a post-processing is adopted to drop duplicate predictions.


### Steps for detection
1. Install `Anaconda 3`
1. Create environment   
    `conda env create -f mmlab_environment.yaml`
2. Activate the environment  
     `source activate mmlab`
3. Prepare the data needed for training HTC    
    **In each step below, before running scripts, change `kaggle_data_root` in each script to your data root path, where `train.csv`, `unicode_translation.csv` and `train_images/` are.**
    1. Generate json annotation in the COCO format  
        `python word_detection/preparation/transform_Kuzushiji_and_split_val_by_book.py`
    2. Generate the instance segmentation mask for each word, based on [this idea](https://www.kaggle.com/frlemarchand/keras-maskrcnn-kuzushiji-recognition)  
        `python word_detection/preparation/addPolyForSegment.py`
    3. Generate semantic segmentation mask for all words in the whole page, see [this](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc) for more details
        `python word_detection/preparation/segment2cocostuff.py`
3. Add `Kuzushiji Dataset` class into mmdet
    1. Copy `word_detection/kuzushiji_dataset.py` from this repo to `/path/to/mmdetection/mmdet/datasets/kuzushiji_dataset.py`.
4. Train HTC for word detection
    1. Change the mmdetection path to your mmdetection path in the training script `word_detection/train_word_det.sh`
    2. Run the trainning script  
4. Run inference to detect all words in test set  
    `KMP_INIT_AT_FORK=FALSE python -u word_detection/test_word_det.py`
5. Group words to lines, and crop the lines
    1. Generate cropped lines and corresponding labels for line recognition model training:    
        `python lines_crop/build_and_crop_lines.py`  
then the cropped imgs would be in `/path/to/data/root/lines_img_train/` and corresbonding labels are in `/path/to/data/root/lines_label_train.txt`, each line in this file is an img of a cropped line;
    2. Generate cropped lines for test set, based on the word detection result  
        1. Comment line `18` to `23` of script `lines_crop/build_and_crop_lines.py`, and uncomment lines `25` to `30`, (also change these paths to yours);
        2. Run this script:  
        `python lines_crop/build_and_crop_lines.py`
then the cropped line imgs would be saved in `/path/to/data/root/lines_img_test/`, and positions of each line in the page are saved in `/path/to/data/root/lines_pos_test.txt`.
        
