import streamlit as st
from PIL import Image
import tensorflow as tf


st.title("DOG VISION ULTIMATE:tm:")

# Define Class Names
class_names = ['Afghan_hound',
               'African_hunting_dog',
               'Airedale',
               'American_Staffordshire_terrier',
               'Appenzeller',
               'Australian_terrier',
               'Bedlington_terrier',
               'Bernese_mountain_dog',
               'Blenheim_spaniel',
               'Border_collie',
               'Border_terrier',
               'Boston_bull',
               'Bouvier_des_Flandres',
               'Brabancon_griffon',
               'Brittany_spaniel',
               'Cardigan',
               'Chesapeake_Bay_retriever',
               'Chihuahua',
               'Dandie_Dinmont',
               'Doberman',
               'English_foxhound',
               'English_setter',
               'English_springer',
               'EntleBucher',
               'Eskimo_dog',
               'French_bulldog',
               'German_shepherd',
               'German_short-haired_pointer',
               'Gordon_setter',
               'Great_Dane',
               'Great_Pyrenees',
               'Greater_Swiss_Mountain_dog',
               'Ibizan_hound',
               'Irish_setter',
               'Irish_terrier',
               'Irish_water_spaniel',
               'Irish_wolfhound',
               'Italian_greyhound',
               'Japanese_spaniel',
               'Kerry_blue_terrier',
               'Labrador_retriever',
               'Lakeland_terrier',
               'Leonberg',
               'Lhasa',
               'Maltese_dog',
               'Mexican_hairless',
               'Newfoundland',
               'Norfolk_terrier',
               'Norwegian_elkhound',
               'Norwich_terrier',
               'Old_English_sheepdog',
               'Pekinese',
               'Pembroke',
               'Pomeranian',
               'Rhodesian_ridgeback',
               'Rottweiler',
               'Saint_Bernard',
               'Saluki',
               'Samoyed',
               'Scotch_terrier',
               'Scottish_deerhound',
               'Sealyham_terrier',
               'Shetland_sheepdog',
               'Shih-Tzu',
               'Siberian_husky',
               'Staffordshire_bullterrier',
               'Sussex_spaniel',
               'Tibetan_mastiff',
               'Tibetan_terrier',
               'Walker_hound',
               'Weimaraner',
               'Welsh_springer_spaniel',
               'West_Highland_white_terrier',
               'Yorkshire_terrier',
               'affenpinscher',
               'american_bulldog',
               'american_pit_bull_terrier',
               'basenji',
               'basset',
               'beagle',
               'black-and-tan_coonhound',
               'bloodhound',
               'bluetick',
               'borzoi',
               'boxer',
               'briard',
               'bull_mastiff',
               'cairn',
               'chow',
               'clumber',
               'cocker_spaniel',
               'collie',
               'curly-coated_retriever',
               'dhole',
               'dingo',
               'english_cocker_spaniel',
               'flat-coated_retriever',
               'giant_schnauzer',
               'golden_retriever',
               'groenendael',
               'havanese',
               'keeshond',
               'kelpie',
               'komondor',
               'kuvasz',
               'malamute',
               'malinois',
               'miniature_pinscher',
               'miniature_poodle',
               'miniature_schnauzer',
               'otterhound',
               'papillon',
               'pug',
               'redbone',
               'schipperke',
               'scottish_terrier',
               'shiba_inu',
               'silky_terrier',
               'soft-coated_wheaten_terrier',
               'standard_poodle',
               'standard_schnauzer',
               'toy_poodle',
               'toy_terrier',
               'vizsla',
               'whippet',
               'wire-haired_fox_terrier']

# Load model
model = tf.keras.models.load_model("model.h5")

def breed_title(label):
  """
  Transforms breed labels into proper syntax for titles
  """
  return label.replace("_"," ").title()

def load_and_prep_image(filename, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor
  and reshapes it to (img_shape, img_shape, color_channel).
  """
  filename = Image.open(filename)
  img = filename.resize((img_shape,img_shape))
  img = img.convert("RGB")
  img = tf.keras.preprocessing.image.img_to_array(img)
  return img

def pred_and_plot(model, filename, class_names=class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the best guess of predicted class.
  """
  # Save original image for display
  original_img = Image.open(filename)

  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class
  pred_class = class_names[pred.argmax()]

  # Plot the image and predicted class
  st.image(original_img)
  st.write(f"Best Guess is: {breed_title(pred_class)}")


# Set File Uploader
file = st.file_uploader("Upload Dog Image", type=["jpg", "png"])

# Page Behavior
if file is None:
 pass
else:
 pred_and_plot(model=model, filename=file)

