from tkinter import *
from tkinter import filedialog
import tkinter as tk
import PIL
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

root = tk.Tk()

WIDTH = 512
HEIGHT = 512
model = None
img = None
path = 'C:\\Users\\gabri\\Desktop\\'

pic = PIL.Image.new('RGB', (WIDTH, HEIGHT), 'black')
draw_img = ImageDraw.Draw(pic)


def get_image():
    global img
    filename = 'image.png'
    pic.save(path + filename)

    img = image.load_img(path + filename, target_size=(28, 28), color_mode="grayscale")
    
    x = image.img_to_array(img)
    x = x.astype('float32')
    x = x.reshape((1,) + x.shape)
    x /= 255

    return x

def draw(e):
    x, y = e.x, e.y

    if canvas.old_coords:
        x_old, y_old = canvas.old_coords
        canvas.create_line(x, y, x_old, y_old, fill="white", width=7)
        draw_img.line((x, y, x_old, y_old), fill="white", width=35)
    canvas.old_coords = x, y

def reset_coords(e):
    canvas.old_coords = None
    
def clear_canvas():
    global pic, draw_img
    canvas.delete("all")
    del draw_img
    pic = PIL.Image.new('RGB', (WIDTH, HEIGHT), 'black')
    draw_img = ImageDraw.Draw(pic)

def predict():
    global model, img
    x = get_image()

    if model == None:
        print("No model was loaded!")
    else:
        pred = model.predict_classes(x)
        print(pred)
        plt.imshow(img)
        plt.show()

def load_Model():
    global model
    model = load_model(filedialog.askopenfilename())

def train_model():
    global model

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images=training_images.reshape(60000, 28, 28, 1)
    training_images=training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images=test_images/255.0
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', restore_best_weights=True)
    mcp_save = ModelCheckpoint('.mdl_wts_{val_loss:.4f}.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    
    history = model.fit(training_images, training_labels, epochs=10, batch_size=128, validation_split=0.2, callbacks=[earlyStopping, mcp_save], verbose=1)
        
        
    # Plotting the training and validation loss
    import matplotlib.pyplot as plt
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
        

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, background='black')
canvas.pack()
canvas.old_coords = None

root.bind('<B1-Motion>', draw)
root.bind('<ButtonRelease-1>', reset_coords)

Btn_predict = Button(text="Predict", command=predict)
Btn_predict.pack()

Btn_clear = Button(text="Clear", command=clear_canvas)
Btn_clear.pack()

Btn_load = Button(text="Load Model", command=load_Model)
Btn_load.pack()

Btn_train = Button(text="Train Model", command=train_model)
Btn_train.pack()

root.mainloop()
