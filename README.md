# DOG VISION ULTIMATE:tm:

[Huggingface Spaces Demo](https://huggingface.co/spaces/Dromarion/Dog-Breed-Classification)

## About the Project
I like dogs but am no good at identifying breeds so for the good of science I must endeavor to train a machine to identify breeds of dogs with a greater accuracy than I ever could.

## The Dataset
The dataset used in training contains the images of the 120 dog breeds of the Stanford Dogs Dataset and is augmented with 6 additional breeds from the Oxford-IIIT Pet Dataset that were not in the original Stanford Dataset. The breeds in the Stanford Dataset have approximately 150 images per breed while the breeds from the Oxford dataset have approximately 200 per breed. For completeness I've also added 200 images of the Siberian Husky Breed to the dataset downloaded from images.cv.

The breeds from the Oxford Dataset are the American Bulldog, American Pit Bull Terrier, English Cocker Spaniel, Havanese, Scottish Terrier and the Shiba Inu.

## The Method

With Tensorflow I used Feature Extraction with EfficientNetV2B3 as the base model. It was trained using Early Stopping for 37 steps until validation loss stalled at 0.2699. After a day of different experiments including fine tuning all layers of the base model, the final best results were 0.2698 loss and 0.9115 accuracy on evaluation. This ended up being done with all layers off the base model frozen and the only real difference being that the learning rate was adjusted to 0.0001.
