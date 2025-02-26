import tkinter as tk
from tkinter import ttk


class Spreadsheet(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # Create canvas and scrollbars
        self.canvas = tk.Canvas(self, background="white")
        self.frame = tk.Frame(self.canvas, background="white")
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.vsb.pack(side="right", fill="y")
        self.hsb.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")

        self.frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.update_visible_cells)
        self.canvas.bind("<MouseWheel>", self.on_scroll)

        self.data = None
        self.cell_widgets = {}
        self.visible_rows = (0, 0)
        self.visible_cols = (0, 0)

    def SetData(self, matrix):
        """Load the matrix and initialize lazy rendering."""
        self.data = matrix
        self.update_visible_cells()

    def update_visible_cells(self, event=None):
        """Only renders cells that are visible in the viewport."""
        if self.data is None or self.data.size == 0:
            return

        # Get visible region
        viewport_width = self.canvas.winfo_width()
        viewport_height = self.canvas.winfo_height()

        if viewport_width == 1 or viewport_height == 1:
            return  # Avoid unnecessary updates when the window is minimized or too small

        row_height = 25
        col_width = 50

        visible_rows = (max(0, self.canvas.yview()[0] * len(self.data)),
                        min(len(self.data), self.canvas.yview()[1] * len(self.data)))
        visible_cols = (max(0, self.canvas.xview()[0] * len(self.data[0])),
                        min(len(self.data[0]), self.canvas.xview()[1] * len(self.data[0])))

        visible_rows = (int(visible_rows[0]), int(visible_rows[1]))
        visible_cols = (int(visible_cols[0]), int(visible_cols[1]))

        if visible_rows == self.visible_rows and visible_cols == self.visible_cols:
            return  # No need to update if viewport is the same

        self.visible_rows, self.visible_cols = visible_rows, visible_cols
        self.render_visible_cells()

    def render_visible_cells(self):
        """Creates and places only visible cell widgets."""
        for widget in self.frame.winfo_children():
            widget.destroy()  # Remove previous widgets

        for r in range(self.visible_rows[0], self.visible_rows[1]):
            for c in range(self.visible_cols[0], self.visible_cols[1]):
                value = self.data[r][c]
                lbl = tk.Label(self.frame, text=str(value), borderwidth=1, relief="solid", width=10, height=1)
                lbl.grid(row=r, column=c, sticky="nsew")

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_frame_configure(self, event):
        """Adjust scroll region when frame resizes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_scroll(self, event):
        """Handles mouse scroll for better performance."""
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")
        self.update_visible_cells()
