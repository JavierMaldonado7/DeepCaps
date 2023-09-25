import tkinter as tk

class Graph(tk.Frame):
    def __init__(self, parent, data, width=400, height=300):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, width=width, height=height, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # calculate graph dimensions
        x_padding = 30
        y_padding = 20
        graph_width = width - x_padding * 2
        graph_height = height - y_padding * 2

        # draw axes
        self.canvas.create_line(x_padding, y_padding, x_padding, y_padding + graph_height, width=2) # y-axis
        self.canvas.create_line(x_padding, y_padding + graph_height, x_padding + graph_width, y_padding + graph_height, width=2) # x-axis

        # draw tick marks and labels
        x_increment = graph_width // (len(data) + 1)
        max_value = max(data)
        y_increment = graph_height // (max_value + 1)
        for i, value in enumerate(data):
            x = x_padding + (i+1) * x_increment
            y = y_padding + graph_height - value * y_increment
            self.canvas.create_line(x, y_padding + graph_height, x, y_padding + graph_height + 5, width=2) # tick mark on x-axis
            self.canvas.create_text(x, y_padding + graph_height + 10, text=str(i+1), anchor=tk.N) # x-axis label
            self.canvas.create_line(x_padding - 5, y, x_padding, y, width=2) # tick mark on y-axis
            self.canvas.create_text(x_padding - 20, y, text=str(value), anchor=tk.E) # y-axis label

        # draw bars
        bar_width = x_increment // 2
        for i, value in enumerate(data):
            x1 = x_padding + (i+1) * x_increment - bar_width // 2
            y1 = y_padding + graph_height
            x2 = x_padding + (i+1) * x_increment + bar_width // 2
            y2 = y_padding + graph_height - value * y_increment
            self.canvas.create_rectangle(x1, y1, x2, y2, fill='blue')

        self.pack(fill=tk.BOTH, expand=True)