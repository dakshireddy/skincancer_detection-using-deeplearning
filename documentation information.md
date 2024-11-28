# Skin-Cancer-Detection
## Skin Cancer Detection with Metadata using deeplearning techniques   
***
### Abstract
***
Nowadays, Skin cancer became a common disease where every 3 in 100 people
are affecting from skin cancer. Previously doctors used to detect the cancer efficiently, but nowadays they are unable to detect. So, there is a drastic demand for
computer-based detection systems. Usually, computer detects skin cancer from
dermoscopic images by using deep learning techniques. Many researchers used
ML and DL techniques to detect skin cancer and to find accurate results. But they
did not perform well for new images. In this paper, we added CNN model to
predict the skin cancer along with that, we proposed a model called Meta Block
which has metadata along with dermoscopy images of patients that includes all
records of patients that helps in predicting the cancer more accurately. In this
paper, we used two different datasets for skin cancer classification and we used
5 different models and compared the results with previous research results. We
found that by comparing the accuracies are increased by 10% in ISIC 2019 Dataset and in HAM1000 Dataset it is increased by 15%.
***  
### Introduction   
***
According to WHO percentage of people who got effected by skin cancer is increased
to 33% in 2021.Back in 2021, Around 1.2 million people died due to skin cancer all
around world. Where 7.22 lakhs of people are men and 4.76 lakhs of people are
women. Skin cancer is affecting a lot of people from Australia, New Zealand and
Denmark than compared with other countries. Where as in India, most skin cancer
cases are affecting from North India. When human body is exposed to UV rays for
longer time then Basal cells in our humans get affected which is the main cause for
skin cancer and these cells damage the unaffected and active ones which increases the
damage percentage. Generally, Basal cells produce new skin cells when old skin
cells are damaged. But when the skin is affected by skin cancer the Basal cells will
not work properly and will not generate new skin cells. Sooner or later the cancer will
spread throughout the body as there are no new cells produced, which is very dangerous and may lead to death.   

Dermoscopy is an assistant diagnosis method which is done by taking some
pictures with the help of computer systems. We used Deep Learning techniques
instead of Machine learning because in ML it takes just a small amount of dataset but
when it comes for classification it requires more data. And even it takes more time for
training the train set to perform as good as dl but still it cannot give us promising and
accurate results. But here comes deep learning which will overcome all the problems
in ml by using neural network. These neural networks are similar to the neurons
in our brain. In dl we can extract the features from the given input and dl can even
take large amount data for train and we can expect efficient accuracy and results. Now
a days, Deep learning became common approach for image detection. It is widely
used for classification and prediction type of problem. Moreover, a greater accuracy
rate of 97.78% was obtained in when the number of layers in CNN was increased
to 14 layers on dermoscopic images from the ISIC dataset.

In previous research papers, They used CNN for processing the image and
later to get better improvements in accuracy they used 5 models for better image classification and for improvement of accuracy in the models of one of the model is Efficient Net B4 which takes less parameters and provides less accuracy we also have
other types of models like Efficient Net B5 which takes less parameters and give us
better accuracy than Efficient Net B4 so in this paper we used Efficient Net B5 as one
of the model in the 5 models which we used.

In our paper, we used CNN instead of ANN because ANN works as human brain
by taking weights. If it anything gets wrong it goes back and gets updates its weights
and again it will work as human brain. The updating of weights is based on cost func-
tion. Where as in CNN, it just uses layers for filtration and analyses the image input.
The layers that are used in CNN are Convolution, ReLU, Pooling, flattening. In our
paper used different types of CNN models like ResNet, GoogleNet, VGGNet, EfficientNet-B5 but found better accuracy for EfficientNet-B5.   
***
### Use of Metadata
***
While there are many CNN models to detect skin cancer but the problem is the prediction of new image with skin cancer. In these models, we used metadata processing.
When it comes to metadata, it has patient records like age, gender, location of infection, diagnosis of cancer, type of skin cancer to patient etc. We used some standard datasets like HAM10000 and ISIC 2019.   
![image](https://github.com/Pavan9303/Skin-Cancer-Detection-/assets/98643288/434a369f-38bb-4bd4-9520-e2df1a33fa36)   
**Fig. 1.** Working of MetaBlock in which we can see that it is the addition of CNN with
Metadata.   
In Fig. 1 it describes the meta block which is our model which contains image data
which is preprocessed, reshaped into a NumPy array and is stored in variable ‘x’ from
the picture we can observe that it also contains metadata which contains records of
patients along with target variable (skin cancer classifier) and we have taken target
variable as ‘y’ which we later categorized into different categories based on skin cancer types in their respective datasets. And used x and y to train and test the CNN model. The overall process will happen inside MetaBlock.    
- Initially imported the dataset from Kaggle and downloaded.
- And Used Label encoder to label the data in the target variable which is in
metadata(CSV file).   
- We resized the image data and converted them into NumPy arrays.
- Next we reshaped the NumPy array by dividing them with 255.
- Then we categorized the target variable into 2D array based on number of classes.
- Later we aggregated features of the image along with target variable.
- Then we applied CNN model, GoogleNet, VGGNet, RESNet, EfficientNet-B5 to the aggregated features.  Finally compared the accuracies for all architectures.
***
### Methodology
***
### CNN (Convolutional Neural Networks)   
CNN is a type of deep learning technique used for image classification and for pixel
classification data. For identifying and recognition CNN is the best architecture. The
Another type of neural network that can uncover key information in both time series
and image data. CNN is also used to identify pattern recognition in the image.   

As we discussed earlier, from convolution to fully connected layer the complexity
of CNN increases. Convolution is the core part of CNN architecture. In this we do a
dot product of image pixels and filter which gives an output know as feature map or
convolved map[1]. Ultimately image is converted into numerical values in this stage
which helps in extracting image patterns. After this ReLU is applied to it. ReLU helps
in converting the negative values into zero and keeping the positive values as it is. In
pooling layer, we reduce the parameters in the input and some information is lost. But
leaving the disadvantage, this helps to reduce complexity of layer and improves effi-
ciency of CNN.

Fully Connected network does the image classification based on features that are
extracted from previous layers. In this our data is flattened into a single array of line
and these are linked to each other. This linked data extracts each feature from one
layer and finally give us the output i.e., image classification.   

In previous research papers, They used CNN for processing the image and
later to get better improvements in accuracy they used 5 models for better image clas-
sification and for improvement of accuracy in the models of one of the model is Efficient Net B4 which takes less parameters and provides less accuracy we also have
other types of models like Efficient Net B5 which takes less parameters and give us
better accuracy than Efficient Net B4 so in this paper we used Efficient Net B5 as one
of the model in the 5 models which we used.

As we know that in image classification features will be extracted from input im-
age that is like patterns, based on the image patterns the model will be trained and
tested accordingly. We use other models to extract shape, color and texture of the
input image. The main problem we face in this is to combine the features extracted
from the image and the metadata. So, we use concatenation to link the images and
metadata in single file and we add extracted images data features in the metadata file
according to the respective images.   

Usually, Image data has a greater number of features than the features of metadata
as image data is high dimensional which means it has high resolution. But the metada-
ta contains only textual format of patient’s records which are less complex than image
data. So Normal concatenation of data might not work in all the cases. So better aggregation data gives us more accurate results.   

### MetaBlock   
Now let’s see The mathematical representation of MetaBlock
Tmetadata = φ(Patients records) 
Timages = φ(Dermoscopy images) 
When we combine both metadata and dermoscopy images we will get MetaBlock
Tmetablock= φ(Patients record + Dermoscopy images) 
Metablock will enchance the feature extraction from the dermoscopy images.   

### GOOGLE NET
Google Net was proposed by Google in 2014 to dig deeper into convolutional
networks. This architecture is very different from other architectures. It is a deep
convolutional neural network with 22 layers (27 layers including pooling layers),
some of these layers are total 9 Inception modules. 224 x 224 dimensions of image is taken as input layer for this architecture. This architecture consists of many things,
like type, patch size, stride, output size, input size, depth, pools, params, ops. In
convolution’s filters 1x1, 3x3, 5x5 are used in the inception module. Whereas, 1x1 is
the filters dimensions.   

### RESNET
Resnet is also known as Residual Network. Residual Network architecture is also
one of the techniques to increase models working efficiency. Used to skip
shifts without affecting model performance. On comparing to other architectures, it
has very deep network (152 layers). Also, this is subject to both vanishing and exploding gradients, impacting model performance when transitioning to the next
epoch. This leaves a learning framework to facilitate much deeper training networks[19]. With ResNet, you can train hundreds or even thousands of layers and still get convincing performance.   

Here we see that ResNet consists of convolution and pooling steps followed
by four layers with similar behavior[5]. Each layer follows the same pattern. They
perform 3x3 convolutions with fixed feature map dimensions (F) [64, 128,
256, 512], bypassing the input every two convolutions. Also, the width (W) and
height (H) dimensions are constant throughout the shift.   

### VGG 16
VGG stands for Visual Geometry Group and is a stand-
ard multilayer deep convolutional neural network (CNN) architecture. This
VGG network consists of 140 million parameters. The 16 in VGG16 refers to 16
layers with weights. VGG16 has 13 convolution layers, 5 max-pooling layers, 3 dense
layers and 21 layers in total, but only 16 weight layers. The input to the network is an
image of dimension (224, 224, 3). The first two layers have 64 channels with a filter
size of 3*3 and the same padding. Then, after a maximum pool layer of stride (2,2), there are two layers of convolution layers with filter size 128 and filter size (3,3).

### EFFICIENT NET B-5
Efficient Net is a powerful Convolution Neural Network to increase model’s performance. This model does Scaling, Depth balancing and Resolution balancing which
leads to efficient output. It has 3*3 and 5*5 convolutional layers and refers to the
various convolutional filters used in the Inception module. It optimizes both accuracy
and efficiency, measured on a floating-point operations per second (FLOPS) basis.
This developed architecture uses mobile reverse bottleneck convolution (MBConv).
***
### Dataset
***
### HAM 10000
The HAM10000 dataset, a large collection of multi-sources dermoscopic images
of common pigmented skin lesions[3]. This dataset contains 10014 dermoscopic
images of 7 different classes namely melanoma, melanocytic nevus, basal cell carcinoma, actinic keratosis, benign keratosis, dermatofibroma, and vascular lesion.
The target values and metadata are present in train.csv file. Size of images is
450*600 pixels.  

![image](https://github.com/Pavan9303/Skin-Cancer-Detection-/assets/98643288/2dc33d37-7dee-462a-b5cd-a334d3b2e49a)   
**Fig. 2.** Samples of skin cancer in HAM10000 dataset   

### ISIC 2019   
The ISIC 2019 dataset, a large collection of multi-sources dermoscopic images
of common pigmented skin lesions[3]. The dataset for ISIC 2019 contains 25,331
images available for the classification of dermoscopic images among nine different
diagnostic categories: Melanoma, Melanocytic nevus, Basal cell carcinoma. The
target values and metadata are present in train.csv file[2]. The size of each image in
the dataset is 512*512 pixels.   

![image](https://github.com/Pavan9303/Skin-Cancer-Detection-/assets/98643288/b01d7b23-a0da-427a-bd0d-aef2a45e684a)   
**Fig. 3.** Samples of skin cancer in ISIC 2019 dataset

***
### Results
***   

![image](https://github.com/Pavan9303/Skin-Cancer-Detection-/assets/98643288/1954ecce-4940-4067-9dae-271fbf7b32a8)

