import tkinter as tk
from solar_analysis_app import SolarAnalysisApp

if __name__ == '__main__':
    # Create the main window (root) for the application using Tkinter
    root = tk.Tk()

    # Initialize the SolarAnalysisApp class, passing the root window as an argument
    app = SolarAnalysisApp(root)

    # Start the Tkinter event loop, which keeps the application running and responsive
    root.mainloop()