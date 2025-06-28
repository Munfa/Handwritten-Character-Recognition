'''
    creating separate files for cleaner code 
    combining all functions in this main.py file
    importing the functions from other files
'''

from data_preprocessing import load_data, map_labels, preprocess
from view_image import show_images
from model import create_model

#### load the dataset
train, test, info = load_data()

#### view sample images before preprocessing 
# show_images(train, "Before Preprocessing") 

#### map the character labels
char_labels = map_labels(train, test)

#### preprocess the data
pre_train, pre_test = preprocess(train, test, info)

#### images after preprocessing
show_images(pre_train.unbatch(), "After Preprocessing", char_labels)
# model = create_model()

# model.fit(
#     train,
#     epochs=10,
#     validation_data=test
# )
# test_loss, test_acc = model.evaluate(test)
# print("Test Accuracy: ", test_acc)