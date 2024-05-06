# MultiFlare

The goal of this project is to classify images into different classes. There are 29 classes in total.

![image](https://github.com/shanunrandev123/MultiFlare/assets/49170258/45ec285c-d8d3-4831-bec3-a8410e639bec)

![image](https://github.com/shanunrandev123/MultiFlare/assets/49170258/403f2f97-01b4-42a9-8d79-fe449bb38bdb)


## Employed techniques to handle multi label Imbalance problem
1. Using a Weighted Loss Function (Weighted BCEWithLogitLoss Function)
2. Using Data Augmentation and create more sample images for minority classes
3. Use Clustering and MLsmote


## Count Distribution of positive(occurence of class) and negative(non occurence of class)
![image](https://github.com/shanunrandev123/MultiFlare/assets/49170258/3a6a0744-ce35-487d-8251-2c53f6459fad)

## Applying weights based on occurence and non-occurence
![image](https://github.com/shanunrandev123/MultiFlare/assets/49170258/7d28e676-07cd-4bee-a4d0-803f335a1002)

![image](https://github.com/shanunrandev123/MultiFlare/assets/49170258/4c569813-68cc-4089-93d4-239c1ca357eb)

![image](https://github.com/shanunrandev123/MultiFlare/assets/49170258/34b4467a-b3ce-4c25-866b-222ce10c0b98)

## Custom Loss function - This can be found in fresh_shanun.py
![image](https://github.com/shanunrandev123/MultiFlare/assets/49170258/fa77fde2-69c6-4aae-840c-80881b8090e9)




