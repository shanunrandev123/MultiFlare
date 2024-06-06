# MultiFlare

## The Dataset provided has an excel sheet with columns as image_id, split_type(train or test), Target class(class 1 to class 29) which is the binarized conversion of the target Very similar to One-hot encoding

### The Training dataset has 64674 images and test dataset has 7186 images. All the preprocessing is done on the training dataset and validation is done on the test dataset. The Training dataset is highly imbalanced and images have different sizes (1024x1024x3, 475x475x3) etc. Resizing is performed on the images to a standard size of 400 x 400 x 3

## dataframe snippet

![image](https://github.com/shanunrandev123/MultiFlare/assets/49170258/b37a15bd-1540-4fdd-862c-69860652ec29)

## Histogram showing the level of imbalance

![image](https://github.com/shanunrandev123/MultiFlare/assets/49170258/d85d11e5-07ae-42d9-bfff-02e38866857a)




