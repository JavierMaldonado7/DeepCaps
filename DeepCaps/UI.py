import sys
import tkinter as tk
import warnings
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
from matplotlib.backends._backend_tk import FigureCanvasTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy.physics.control.control_plots import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import CapsuleNet
import PlotResults
import Routing
import Train
import main
from GradientFrame import GradientFrame
from tkinter import *
from PIL import Image, ImageTk


class ConsoleWriter:
    def __init__(self, console, tag):
        self.console = console
        self.tag = tag

    def write(self, message):
        self.console.configure(state="normal")
        self.console.insert("end", message, (self.tag,))
        self.console.configure(state="disabled")

    def flush(self):
        pass

class App:
    def __init__(self, master):
        self.master = master
        self.testing_file = ""
        self.testing_button = tk.Button(master, bg="#FE9677", font=("Roboto", 12, "bold"), padx=5, pady=5, text="Select Testing Set  ", command=self.select_testing_file)
        self.testing_button.grid(row=0, column=0, padx=0, pady=0)

        self.g4 = GradientFrame(master, colors=("#984063", "#FE9677"), width=200, height=1000,  highlightthickness=0)
        self.g4.config(direction=self.g4.top2bottom)
        self.g4.grid(row=2, column=0, rowspan=4, sticky="news", padx=0, pady=0)

        # Create the button to detect
        self.detect_button = tk.Button(master, bg="#FE9677", font=("Roboto", 12, "bold"), padx=5, pady=5, text="     Detect     ", command=self.detect)
        self.detect_button.grid(row=1, column=0, padx=0, pady=0)

        self.console = tk.Text(master, bg="black", fg="white", font=("Roboto", 12, "bold"), wrap="word", state="disabled", width=100, height=480, padx=0, pady=0 ,  highlightthickness=0)
        self.console.grid(row=3, column=3, padx=0, pady=0)

        sys.stdout = ConsoleWriter(self.console, "stdout")
        sys.stderr = ConsoleWriter(self.console, "stderr")

        self.gf = GradientFrame(master, colors=( "#984063","#FE9677"), width=1700, height=8000,  highlightthickness=0)
        self.gf.config(direction=self.gf.left2right)
        self.gf.grid(row=0, column=5, rowspan=4, sticky="news", padx=0, pady=0)


    def select_testing_file(self):
        # Open the file explorer
        file_path = filedialog.askopenfilename()

        # Add the selected file to the list of testing files
        self.testing_file = file_path



    def process_files(self):
        # Process the selected files (e.g., print their paths)

        main.Train()

    def detect(self):
        root.update()
        route = Routing.Routing
        main.Train()
        print("Detection in progress...")
        main.Detect(route,self.testing_file)



# Create the main window
root = tk.Tk()
root.geometry("1200x800")
root.configure(bg = "#984063")
root.title("DeepCapsNet")
# root.iconbitmap("icon.ico") # set the window icon to "icon.ico"
# Create the app
app = App(root)

# Run the main event loop
root.mainloop()
