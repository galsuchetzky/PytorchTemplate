import tkinter as tk
from tkinter import ttk
from data_loader import BREAKLogical

datasets = ["BREAKLogical"]
splits = ['train', 'evaluation', 'test']


class Gui:
    def __init__(self, root):
        self.root = root
        root.title("Project GUI")

        self.create_tabs()
        self.init_dataset_tab()

    def create_tabs(self):
        self.tabs_controller = ttk.Notebook(root)
        self.dataset_tab = ttk.Frame(self.tabs_controller)
        self.train_tab = ttk.Frame(self.tabs_controller)

        self.tabs_controller.add(self.dataset_tab, text="Dataset")
        self.tabs_controller.add(self.train_tab, text="Train")
        self.tabs_controller.pack()

    def init_dataset_tab(self):
        # Create an option menu for selecting the dataset to work with.
        self.dataset_selection_label = tk.Label(self.dataset_tab, text="dataset:")
        self.dataset_selection_label.grid(row=0, column=0)

        self.dataset_variable = tk.StringVar(self.dataset_tab)
        self.dataset_variable.set("BREAKLogical")  # default value
        self.datasets_option_menu = tk.OptionMenu(self.dataset_tab, self.dataset_variable, *datasets)
        self.datasets_option_menu.grid(row=0, column=1)

        # Create split selection option menu.
        self.dataset_split_label = tk.Label(self.dataset_tab, text="split:")
        self.dataset_split_label.grid(row=0, column=2)

        self.dataset_split_variable = tk.StringVar(self.dataset_tab)
        self.dataset_split_variable.set("train")  # default value
        self.split_option_menu = tk.OptionMenu(self.dataset_tab, self.dataset_split_variable, *splits)
        self.split_option_menu.grid(row=0, column=3)

        # Create a load button for the dataset.
        self.load_dataset_button = tk.Button(self.dataset_tab, text="Load", command=self.load_dataset)
        self.load_dataset_button.grid(row=0, column=4)

    def load_dataset(self):
        print("loading dataset")
        if self.dataset_variable.get() == 'BREAKLogical':
            training = False
            evaluation = False
            if self.dataset_split_variable.get() == "training":
                training = True
            if self.dataset_split_variable.get() == "evaluation":
                evaluation = True
            self.dataset = BREAKLogical('../data/', train=training, valid=evaluation)
        print("done loading dataset")



if __name__ == '__main__':
    root = tk.Tk()
    gui = Gui(root)
    root.mainloop()
