import tkinter as tk
import os
import threading
import numpy as np
from tkinter import ttk, messagebox
from cls_Generate import generate_H, generate_m
from spreadsheet import Spreadsheet  # Import the Spreadsheet widget
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ISDApp(tk.Tk):
    default_n = 200
    default_k = 100
    default_t = 5
    last_n = ""
    last_k = ""
    last_t = ""

    def __init__(self):
        super().__init__()
        self.title("Information Set Decoding")
        self.state("zoomed")

        # Main container panel
        self.pnl_main = ttk.Frame(self)
        self.pnl_main.pack(fill=tk.BOTH, expand=True)
        self.pnl_main.columnconfigure(0, weight=1, uniform="equal")
        self.pnl_main.columnconfigure(1, weight=1, uniform="equal")
        self.pnl_main.rowconfigure(0, weight=1)

        # Left panel for initialization and matrix / m-vector / y-list display
        self.pnl_left = ttk.Frame(self.pnl_main, padding=0)
        self.pnl_left.grid(row=0, column=0, sticky="nsew")
        self.pnl_left.columnconfigure(0, weight=1)
        self.pnl_left.rowconfigure(1, weight=0)
        self.pnl_left.rowconfigure(2, weight=0)
        self.pnl_left.rowconfigure(3, weight=1)
        self.pnl_left.rowconfigure(4, weight=0)

        # Initialization panel
        self.pnl_init = ttk.Frame(self.pnl_left, padding=10)
        self.pnl_init.grid(row=0, column=0, sticky="ew")
        self.pnl_init.columnconfigure(0, weight=1)

        # Panel for option selection
        self.pnl_options = tk.Frame(self.pnl_init)
        self.pnl_options.grid(row=0, column=0, sticky="ew")
        self.pnl_options.columnconfigure(0, weight=1)

        # Option list for initialization method
        self.sel_init = tk.StringVar(value="load_last")
        self.sel_init.trace_add("write", self.on_option_change)
        self.opt_test = tk.Radiobutton(self.pnl_options,
                                       variable=self.sel_init,
                                       value="load_test",
                                       text="Load test matrix and m-vector")
        self.opt_test.pack(anchor="w")
        self.opt_load = tk.Radiobutton(self.pnl_options,
                                       variable=self.sel_init,
                                       value="load_last",
                                       text="Load last generated matrix and m-vector")
        self.opt_load.pack(anchor="w")
        self.opt_generate = tk.Radiobutton(self.pnl_options,
                                           variable=self.sel_init,
                                           value="generate",
                                           text="Generate matrix and m-vector (change initialization values if needed)")
        self.opt_generate.pack(anchor="w")

        # Button for initialisation
        self.init_button = tk.Button(self.pnl_init,
                                     text="Initialize",
                                     width=15,
                                     command=self.initialize)
        self.init_button.grid(row=0, column=1, sticky="nsew")

        # Panel for values of n, k and t
        self.pnl_input = tk.Frame(self.pnl_left)
        self.pnl_input.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.pnl_input.columnconfigure((1, 3, 5), weight=1)
        tk.Label(self.pnl_input, text="n =").grid(row=0, column=0, padx=(5, 0))
        self.n_entry = ttk.Entry(self.pnl_input, state="readonly")
        self.n_entry.grid(row=0, column=1, sticky="ew")
        tk.Label(self.pnl_input, text="k =").grid(row=0, column=2, padx=(10, 0))
        self.k_entry = ttk.Entry(self.pnl_input, state="readonly")
        self.k_entry.grid(row=0, column=3, sticky="ew")
        tk.Label(self.pnl_input, text="t =").grid(row=0, column=4, padx=(10, 0))
        self.t_entry = ttk.Entry(self.pnl_input, state="readonly")
        self.t_entry.grid(row=0, column=5, sticky="ew")

        # Panel for displaying input data (pnl_progress_data) with border and notebook-like background
        style = ttk.Style()
        notebook_bg = style.lookup("TNotebook", "background")
        self.pnl_progress_data = tk.Frame(self.pnl_left, background=notebook_bg, bd=1, relief="solid")
        self.pnl_progress_data.grid(row=3, column=0, sticky="nsew", pady=0)
        self.pnl_progress_data.columnconfigure(0, weight=1)
        self.pnl_progress_data.rowconfigure(0, weight=1)
        self.pnl_progress_data.rowconfigure(1, weight=0)

        # Top frame inside pnl_progress_data for tab_H and tab_y
        self.top_frame = tk.Frame(self.pnl_progress_data, background=notebook_bg)
        self.top_frame.grid(row=0, column=0, sticky="nsew")
        self.top_frame.rowconfigure(0, weight=1)
        self.top_frame.columnconfigure(0, weight=1)
        self.top_frame.columnconfigure(1, weight=0)

        # Create tab_H notebook
        self.tab_H = ttk.Notebook(self.top_frame)
        self.tab_H.grid(row=0, column=0, sticky="nsew")
        self.tab_H_page = ttk.Frame(self.tab_H)
        self.tab_H.add(self.tab_H_page, text="matrix (H)")

        # Add Spreadsheet widget inside tab_H_page
        self.spreadsheet_H = Spreadsheet(self.tab_H_page)
        self.spreadsheet_H.pack(fill=tk.BOTH, expand=True)

        # Create tab_y notebook with fixed width
        self.tab_y = ttk.Notebook(self.top_frame, width=200)
        self.tab_y.grid(row=0, column=1, sticky="nsew")
        self.tab_y_page = ttk.Frame(self.tab_y)
        self.tab_y.add(self.tab_y_page, text="y")

        # Bottom frame inside pnl_progress_data for tab_m
        self.bottom_frame = tk.Frame(self.pnl_progress_data, background=notebook_bg, height=150)
        self.bottom_frame.grid(row=1, column=0, sticky="ew")
        self.bottom_frame.grid_propagate(False)
        self.tab_m = ttk.Notebook(self.bottom_frame)
        self.tab_m.pack(fill=tk.BOTH, expand=True)
        self.tab_m_page = ttk.Frame(self.tab_m)
        self.tab_m.add(self.tab_m_page, text="Vector (m)")

        # Initially, make tab_H, tab_y and tab_m invisible
        #self.top_frame.grid_remove()
        #self.bottom_frame.grid_remove()

        # Right panel for tabbed results
        self.pnl_right = ttk.Frame(self.pnl_main, padding=10)
        self.pnl_right.grid(row=0, column=1, sticky="nsew")
        self.algorithm_tabs = ttk.Notebook(self.pnl_right)
        self.algorithm_tabs.pack(fill=tk.BOTH, expand=True)
        self.algorithm_1_tab = ttk.Frame(self.algorithm_tabs)
        self.algorithm_tabs.add(self.algorithm_1_tab, text="Algorithm 1")
        self.algorithm_2_tab = ttk.Frame(self.algorithm_tabs)
        self.algorithm_tabs.add(self.algorithm_2_tab, text="Algorithm 2")
        self.algorithm_3_tab = ttk.Frame(self.algorithm_tabs)
        self.algorithm_tabs.add(self.algorithm_3_tab, text="Algorithm 3")

        # Progress indicator within pnl_progress_data (overlayed on top of the notebooks area)
        self.progress_label = tk.Label(self.pnl_progress_data, text="",
                                       font=("Arial", 12))
        self.progress_spinner = ttk.Progressbar(self.pnl_progress_data, mode="indeterminate")
        self.progress_label.place(relx=0.5, rely=0.4, anchor="center")
        self.progress_spinner.place(relx=0.5, rely=0.6, anchor="center")
        self.progress_label.place_forget()
        self.progress_spinner.place_forget()
        self.progress_spinner.stop()

    def on_option_change(self, *args):
        if self.sel_init.get() == "generate":
            self.n_entry.config(state="normal")
            self.k_entry.config(state="normal")
            self.t_entry.config(state="normal")

            init_values = {}
            if os.path.exists("init.dat"):
                with open("init.dat", "r") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line:
                            key, value = map(str.strip, line.split("=", 1))
                            if key in {"n", "k", "t"}:
                                init_values[key] = value

            if not self.n_entry.get():
                self.n_entry.delete(0, tk.END)
                self.n_entry.insert(0, init_values.get("n", str(self.default_n)))
            if not self.k_entry.get():
                self.k_entry.delete(0, tk.END)
                self.k_entry.insert(0, init_values.get("k", str(self.default_k)))
            if not self.t_entry.get():
                self.t_entry.delete(0, tk.END)
                self.t_entry.insert(0, init_values.get("t", str(self.default_t)))
        else:
            self.n_entry.delete(0, tk.END)
            self.n_entry.insert(0, self.last_n)
            self.n_entry.config(state="readonly")
            self.k_entry.delete(0, tk.END)
            self.k_entry.insert(0, self.last_k)
            self.k_entry.config(state="readonly")
            self.t_entry.delete(0, tk.END)
            self.t_entry.insert(0, self.last_t)
            self.t_entry.config(state="readonly")

    def show_progress(self, text):
        self.progress_label.config(text=text)
        self.progress_label.place(relx=0.5, rely=0.4, anchor="center")
        self.progress_spinner.place(relx=0.5, rely=0.6, anchor="center")
        self.progress_spinner.start(5)

    def hide_progress(self):
        self.progress_label.place_forget()
        self.progress_spinner.place_forget()
        self.progress_spinner.stop()

    def initialize(self):
        self.show_progress("Generating matrix (H) and vector (m)...")
        threading.Thread(target=self._process_initialization, daemon=True).start()

    def _process_initialization(self):
        option = self.sel_init.get()
        match option:
            case "load_test":
                # Implement test option functionality here.
                # For now, simply inform the user.
                self.after(0, self.hide_progress)
                self.after(0, lambda: messagebox.showinfo("Info", "Test option selected."))
                return
            case "load_last":
                # Implement load option functionality here.
                # For now, simply inform the user.
                self.after(0, self.hide_progress)
                self.after(0, lambda: messagebox.showinfo("Info", "Load option selected."))
                return
            case "generate":
                try:
                    n = int(self.n_entry.get())
                    k = int(self.k_entry.get())
                    t = int(self.t_entry.get())
                    if t > n:
                        messagebox.showerror("Input Error", "Hamming weight 't' cannot be greater than 'n'.")
                        self.after(0, self.hide_progress)
                        return
                    if k > n:
                        messagebox.showerror("Input Error",
                                             "In a matrix having full row rank, the number of rows (k) cannot be greater than number of columns (n).")
                        self.after(0, self.hide_progress)
                        return
                    cnt = [0]
                    self.after(0, self.show_progress, "Generating matrix (H)...")
                    H = generate_H(n, k)
                    self.after(0, self.show_progress, "Generating vector (m)...")
                    m = generate_m(n, t)
                    self.after(0, self.show_progress, "Saving generated matrix (H) and vector (m)...")
                    self.save_H_m(H, m)
                    self.after(0, lambda: self.spreadsheet_H.SetData(H))  # Assign H to the Spreadsheet widget
                    self.after(0, self.hide_progress)
                    #self.after(0, self.top_frame.grid)
                    #self.after(0, self.bottom_frame.grid)
                except ValueError:
                    self.after(0, lambda: messagebox.showerror("Input Error", "Values of 'n', 'k', and 't' must be non-empty integer values."))
            case _:
                self.after(0, self.hide_progress)
                return

        #self.after(0, self.top_frame.grid)
        #self.after(0, self.bottom_frame.grid)

    def save_H_m(self, H, m):
        with open("generate.dat", "w") as f:
            for row in H:
                f.write(",".join(map(str, row)) + "\n")
            f.write("\n")  # Empty line
            f.write(",".join(map(str, m)) + "\n")

if __name__ == "__main__":
    app = ISDApp()
    app.mainloop()
