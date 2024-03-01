from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QFont
from keras.models import load_model
import keras.utils as image
import sys
import numpy as np

class CancerClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Cancer Classifier")
        self.setGeometry(100, 100, 1024, 768)  # Set initial window size to 1024x768
        self.setFixedSize(1024, 768)  # Fix the window size

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        
        #self.blank = QLabel('',self)
        #self.blank.setStyleSheet("border-radius: 10px; padding: 10px;")
        #self.blank.setGeometry(0, 0, 1024, 768)

        self.blank = QLabel('',self)
        self.blank.setStyleSheet("border-radius: 10px; padding: 10px;")
        self.blank.setGeometry(0, 0, 276, 80)

        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.show_file_dialog)
        self.upload_button.setGeometry(10, 10, 256, 60)
        self.upload_button.setStyleSheet("background-color: #0a0a0a; color: white; border-style: outset; border-width: 2px; border-radius: 10px; border-color: beige; font: bold 14px; min-width: 10em; padding: 6px;")

        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        self.central_widget.setLayout(self.layout)

        self.model = load_model("Model.h5")
        self.class_names = ['a colon adenocarcinoma', 'colon cells with no cancer', 'a lung adenocarcinoma',
                            'lung cells with no cancer', 'a lung squamous cell carcinoma']

        # Set background color
        self.setStyleSheet("background-color: #1f1f1f;")

    def show_file_dialog(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            self.display_image(file_path)

    def display_image(self, file_path):
        img = image.load_img(file_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        pred = self.model.predict(img)
        classes = np.argmax(pred, axis=1)
        class_name = self.class_names[classes[0]]
        confidence = round(max(max(pred)) * 100, 2)

        output_text = f"It's {class_name}\nConfidence: {confidence}%"

        pixmap = QPixmap(file_path)
        self.label.setPixmap(pixmap.scaledToWidth(1024))  # Scale the image to fit the fixed width of 1024
        self.label.setScaledContents(True)

        result_label = QLabel(output_text, self)
        result_label.setStyleSheet("color: white; font-size: 32px; background-color: red; border-radius: 10px; padding: 10px;")  # Set text color to white, increase font size, and add red box
        self.layout.addWidget(result_label)

def on_close():
    app.quit()
    sys.exit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = CancerClassifierApp()
    mainWin.show()

    app.aboutToQuit.connect(on_close)  # Connect the aboutToQuit signal to the on_close function

    sys.exit(app.exec_())

