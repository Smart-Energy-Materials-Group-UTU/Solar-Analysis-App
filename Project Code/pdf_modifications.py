import os
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
import pandas as pd
from file_processing import TwoPixelSampleAnalyzer, TwoPixelMeasurementAnalyzer, EightPixelDataAnalyzer, EightPixelSampleAnalyzer
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

# Define the base directory
base_dir = os.getcwd()

# Define the paths for Reports folders
reports_dir = os.path.join(base_dir, "Reports")
icon_dir = os.path.join(base_dir, "Images")

# Ensure Reports and Images directory exists
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(icon_dir, exist_ok=True)

class TwoPixelReportGenerator:
    """
    Generates a PDF report for 2-pixel solar cell measurements.

    Responsibilities:
    - Load and analyze J–V data from a single Excel file.
    - Calculate key photovoltaic parameters (PCE, FF, Jsc, Voc, HI).
    - Render cover page, per-sheet result pages (graphs + tables), and a final summary page.
    - Insert institutional and experiment metadata, page numbers, and logos.
    """
      
    def __init__(self, file_path, experimenter, metadata):

        self.file_path = file_path # Path to source Excel file
        self.experimenter = experimenter # Name to appear on report
        self.analyzed = TwoPixelMeasurementAnalyzer(file_path) # Data analyzer instance
        self.timestamp = datetime.now().strftime("%H%M%S_%d%m%Y") # Timestamp for unique filename
        self.pdf_filename = f"2-Pixel_data_report_{self.timestamp}_{self.experimenter}.pdf" # Constructed output PDF name
        self.pdf_path = os.path.join(reports_dir, self.pdf_filename) # Full output path in Reports directory
        self.current_page_number = 1  # Initialize page counter
        self.pce_values = []  # Store PCE values for boxplot
        self.c = canvas.Canvas(self.pdf_path, pagesize=A4) # ReportLab canvas for PDF drawing
        self.metadata = metadata # Dict of experiment metadata

    def _add_page_number(self):
        """Adds the page number to the bottom of the current page."""
        self.c.setFont("Helvetica", 10)  # Set font for page number
        self.c.setFillColor(colors.grey)  # Set text color to light gray
        page_number_text = f"Page {self.current_page_number}"
        text_width = self.c.stringWidth(page_number_text, "Helvetica", 10)
        self.c.drawString(A4[0] - text_width - 20, 30, page_number_text)  # Position at bottom center
        self.current_page_number += 1  # Increment page counter
        self.c.setFillColor(colors.black)  # Reset text color to black (optional)
        
    def _add_icons(self, opacity = 1, scale = 1): # 'opacity' parameter to set transparency level (1.0 = full, 0.0 = invisible), 'scale' parameter to aspect ratio
        '''Adds the institutional icons'''
        # Add the first icon (left)
        icon_left_path = os.path.join(icon_dir, "UTU_logo_EN_RGB.png")  # Replace with your left icon path
        if os.path.exists(icon_left_path):
            left_icon_width = 165.6
            left_icon_height = 61.6  # Adjusted to maintain aspect ratio (828x308)
            margin = 50  # Consistent margin from the edges
            self.c.setFillAlpha(opacity)
            self.c.drawImage(icon_left_path, x=margin, y=A4[1] - left_icon_height*scale - margin-15, 
                            width=left_icon_width*scale, height=left_icon_height*scale, mask='auto')
        else:
            print(f"Warning: Left icon not found at {icon_left_path}")

        # Add the second icon (right)
        icon_right_path = os.path.join(icon_dir, "smat_logo_png.png")  # Replace with your right icon path
        if os.path.exists(icon_right_path):
            right_icon_width = 50.1
            right_icon_height = 63.83  # 301x383
            # Align the left edge of the right icon with the right margin
            self.c.drawImage(icon_right_path, x=A4[0] - right_icon_width*scale - margin, 
                            y=A4[1] - right_icon_height*scale - margin - 15, 
                            width=right_icon_width*scale, height=right_icon_height*scale, mask='auto')    
        else:
            print(f"Warning: Right icon not found at {icon_right_path}")

    def _add_cover_page(self):
        """Creates the cover page with logos and title."""

        self.c.setFont("Helvetica-Bold", 24)
        text = "Solar Cell Measurement Results"
        text_width = self.c.stringWidth(text, "Helvetica-Bold", 24)
        self.c.drawString((A4[0] - text_width) / 2, A4[1] / 2, text)

        # Add experimenter's name below the title
        self.c.setFont("Helvetica", 15)  # Slightly smaller font
        experimenter_text = f"Experimenter: {self.experimenter}"
        experimenter_width = self.c.stringWidth(experimenter_text, "Helvetica", 15)
        self.c.drawString((A4[0] - experimenter_width) / 2, A4[1] / 2 - 30, experimenter_text)
        
        # Retrieve metadata
        metadata_text = []
        if self.metadata.get("Scan Rate"):
            metadata_text.append(f"Scan Rate: {self.metadata['Scan Rate']} V/s")
        if self.metadata.get("Sun Intensity"):
            metadata_text.append(f"Sun Intensity: {self.metadata['Sun Intensity']} W/m²")
        if self.metadata.get("Temperature"):
            metadata_text.append(f"Temperature: {self.metadata['Temperature']} °C")

        # Display the 'Measurement Conditions' text
        self.c.setFont("Helvetica", 13)
        text_width = self.c.stringWidth('Measurement Conditions:', "Helvetica", 13)
        self.c.drawString((A4[0] - text_width) / 2, A4[1] / 2 - 70, 'Measurement Conditions:')

        # Display metadata above the experiment date
        self.c.setFont("Helvetica", 12)
        metadata_y = A4[1] / 2 - 90  # Start below the experimenter name

        for line in metadata_text:
            metadata_width = self.c.stringWidth(line, "Helvetica", 12)
            self.c.drawString((A4[0] - metadata_width) / 2, metadata_y, line)
            metadata_y -= 20  # Move down for the next line
        
        # Experiment date
        date_text = f'Experiment date: {self.metadata["Experiment Date"]} | Report Generated: {datetime.now().strftime("%d/%m/%Y")}'
        self.c.setFont("Helvetica", 12)
        date_width = self.c.stringWidth(date_text, "Helvetica", 12)
        self.c.drawString((A4[0] - date_width) / 2, 40, date_text)

        self._add_icons()
        self.c.showPage()
        
    def _add_sheet_results(self, sheet_name, graph_count, sample_count, next_sheet_name=None):
        """Processes each sheet and adds its results to the PDF."""
        page_width, page_height = A4  # Get the page dimensions

        # Load and validate JV + performance data for this sheet
        self.processor = TwoPixelSampleAnalyzer(self.file_path, sheet_name, next_sheet_name=next_sheet_name)
        plot = self.processor.generate_jv_graph()  # Using the method from sample_analyzer

        if plot is None:
            print(f'No J-V Plot generated for the {sheet_name}. Skipping the sheet')
            return False

        # Page title
        self.c.setFont("Helvetica-Bold", 14)
        page_title = f"Results of Sample {sample_count}: {'Left' if '-l-' in sheet_name else 'Right'} Pixel"
        text_width = self.c.stringWidth(page_title, "Helvetica-Bold", 14)
        self.c.drawString((page_width - text_width) / 2, page_height - 65, page_title)

        # Graph size and positioning (horizontally centered)
        graph_width = 400  # Adjusted width
        graph_height = 300  # Adjusted height
        graph_x_position = (page_width - graph_width) / 2  # Horizontally center the graph
        graph_y_position = 400  # Adjusted y-position for the graph, this parameter sets the positions of the graphs and tables dynamically

        self.c.drawImage(plot, graph_x_position, graph_y_position, width=graph_width, height=graph_height)

        # Graph title just below the graph
        graph_title = f"Figure {graph_count}. J-V Curve of Sample {graph_count}: {'Left' if '-l-' in sheet_name else 'Right'} Pixel"
        graph_title_width = self.c.stringWidth(graph_title, "Helvetica", 12)
        self.c.setFont("Helvetica", 12)
        self.c.drawString((page_width - graph_title_width) / 2, graph_y_position - 20, graph_title)

        # Add Table Title below the graph title
        table_title = f"Table {graph_count}. Performance Data of Sample {graph_count}: {'Left' if '-l-' in sheet_name else 'Right'} Pixel"
        table_title_width = self.c.stringWidth(table_title, "Helvetica", 12)
        self.c.setFont("Helvetica", 12)
        table_title_height=graph_y_position - 50
        self.c.drawString((page_width - table_title_width) / 2, table_title_height, table_title)

        # Add Table (Horizontally arranged)
        table_data = self.processor.construct_table()

        table_data_transposed = list(zip(*table_data))  # Transpose for dual-scan format

        # Transpose and style performance-data table for dual‐scan format
        table = Table(table_data_transposed)
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),  # Reduce font size to 8
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        table.wrapOn(self.c, page_width - 100, page_height)
        table_width = table._width  # Get the actual width of the table
        table_height = table._height  # Get actual rendered height
        
        # Dynamically adjust vertical spacing
        BUFFER_SPACE = 20  # Space between title and table
        table_y_position = table_title_height - table_height - BUFFER_SPACE
        table_height = table._height  # Get actual rendered height

        table.drawOn(self.c, (page_width - table_width) / 2 , table_y_position)  # Adjusted position for the table

        
        # Calculate and render Hysteresis Index under the table
        hi_value = self.processor.calculate_hi()

        if hi_value is not None:
            # Displaying calculated HI
            self.c.setFont("Helvetica", 12)
            page_title = f"The calculated Hysteresis Index (HI) for {'Left' if '-l-' in sheet_name else 'Right'} Pixel is {hi_value}"
            text_width = self.c.stringWidth(page_title, "Helvetica", 12)
            left_margin = 56.7
            self.c.drawString(left_margin, table_y_position - 30, page_title)

        self._add_page_number()
        self._add_icons(opacity = 0.5, scale=0.7)
        self.c.showPage()
        return True

    def _add_final_page(self, graph_count):
        """Generates the final page with boxplots arranged in a 2x2 grid, with the title below the plots."""
        page_width, page_height = A4  # Get the page dimensions

        # Page title
        self.c.setFont("Helvetica-Bold", 14)
        page_title = "Solar Cell Key Performance Metrics"
        text_width = self.c.stringWidth(page_title, "Helvetica-Bold", 14)
        self.c.drawString((page_width - text_width) / 2, page_height - 65, page_title)

        # Define the size and positions for the plots in the grid
        plot_width = 250
        plot_height = 200
        margin_x = 50
        margin_y = 200
        x_gap = 250  # Horizontal gap between plots
        y_gap = 200  # Vertical gap between plots

        # Plotting PCE Values (Top-left)
        plot = TwoPixelMeasurementAnalyzer.plot_boxplot(self.analyzed.pce.fw, self.analyzed.pce.rv, 'PCE values', 'PCE (%)')
        self.c.drawImage(plot, margin_x, margin_y + y_gap, width=plot_width, height=plot_height)
        self.c.setFont('Helvetica', 10)
        self.c.drawString(margin_x, margin_y + y_gap + 185, "(a)")

        # Plotting Fill Factor Values (Top-right)
        plot = TwoPixelMeasurementAnalyzer.plot_boxplot(self.analyzed.ff.fw, self.analyzed.ff.rv, 'Fill Factor values', 'FF')
        self.c.drawImage(plot, margin_x + x_gap, margin_y + y_gap, width=plot_width, height=plot_height)
        self.c.drawString(margin_x + x_gap , margin_y + y_gap + 185, "(b)")
        # Plotting J sc (Bottom-left)
        plot = TwoPixelMeasurementAnalyzer.plot_boxplot(self.analyzed.jsc.fw, self.analyzed.jsc.rv, 'Short Circuit Current Densities', 'Jsc (mA/cm²)')
        self.c.drawImage(plot, margin_x, margin_y, width=plot_width, height=plot_height)
        self.c.drawString(margin_x, margin_y + 185, "(c)")
        # Plotting V oc (Bottom-right)
        plot = TwoPixelMeasurementAnalyzer.plot_boxplot(self.analyzed.voc.fw, self.analyzed.voc.rv, 'Open Circuit Voltage values', 'Voc (V)')
        self.c.drawImage(plot, margin_x + x_gap, margin_y, width=plot_width, height=plot_height)
        self.c.drawString(margin_x + x_gap, margin_y + 185, "(d)")
        # Add the title below the graphs
        self.c.setFont('Helvetica', 12)
        figure_title = f"Figure {graph_count}. (a) PCE, (b) FF, (c) Jsc, (d) Voc of the samples"
        title_width = self.c.stringWidth(figure_title, 'Helvetica', 12)
        self.c.drawString((page_width - title_width) / 2, margin_y - 15, figure_title)

        # Add page number and icons
        self._add_page_number()
        self._add_icons(opacity=0.5, scale=0.6)

        # Increment graph count for subsequent pages
        graph_count += 1

    def save_pdf(self):
        """Runs the full report generation process."""
        self._add_cover_page()

        xls = pd.ExcelFile(self.file_path)
        sheets = xls.sheet_names # A list of all sheet names in the Excel file

        # Count how many graphs have been generated so far (for report formatting)
        graph_count = 1
        sample_count= 1

        # Iterate through sheet pairs: left and right pixel scans
        for i in range(0, len(sheets), 2):
            current_sheet = sheets[i]
            next_sheet = sheets[i + 1] if i + 1 < len(sheets) else None
            plot_created = self._add_sheet_results(current_sheet, graph_count, sample_count, next_sheet_name=next_sheet)
            
            if plot_created:
                graph_count += 1
            else:
                # Abort if no sheets/data found in the Excel file
                break

            if graph_count % 2 == 1:  # Increment sample_count after processing both left and right pixels
                sample_count += 1
            
        self._add_final_page(graph_count)
        self.c.save()
        print(f"PDF generated: {self.pdf_path}")

class EightPixelReportGenerator:
    """
    Generates a PDF report for 8-pixel solar cell measurements.

    Responsibilities:
    - Iterate over each sample and its pixels, loading JV and performance CSVs.
    - Analyze each pixel’s data, generate J-V plots, tables, and calculate Hysteresis Index (HI).
    - Render a cover page, per-pixel result pages, and a final box-plot summary page.
    - Insert page numbers and institutional logos on every page.
    """
    
    def __init__(self, jv_file_paths, experimenter):
        self.jv_file_paths = jv_file_paths # dict: {sample_name: {pixel_number: [jv_csv, perf_csv, …]}}
        self.experimenter = experimenter # Name of the researcher
        self.timestamp = datetime.now().strftime("%H%M%S_%d%m%Y")  # Unique timestamp for filename
        self.pdf_filename = f"8-Pixel_data_report_{self.timestamp}_{self.experimenter}.pdf" # Output PDF filename pattern
        self.pdf_path = os.path.join(reports_dir, self.pdf_filename) # Full path under `reports_dir`
        self.current_page_number = 1  # Initialize page counter
        self.pce_values = []  # Store PCE values for boxplot
        self.c = canvas.Canvas(self.pdf_path, pagesize=A4) # ReportLab Canvas for PDF drawing

    def _add_page_number(self):
        """Adds the page number to the bottom of the current page."""
        self.c.setFont("Helvetica", 10)  # Set font for page number
        self.c.setFillColor(colors.grey)  # Set text color to light gray
        page_number_text = f"Page {self.current_page_number}"
        text_width = self.c.stringWidth(page_number_text, "Helvetica", 10)
        self.c.drawString(A4[0] - text_width - 20, 30, page_number_text)  # Position at bottom center
        self.current_page_number += 1  # Increment page counter
        self.c.setFillColor(colors.black)  # Reset text color to black (optional)
        
    def _add_icons(self, opacity = 1, scale = 1): # 'opacity' parameter to set transparency level (1.0 = full, 0.0 = invisible), 'scale' parameter to aspect ratio   
        # Add the first icon (left)
        icon_left_path = os.path.join(icon_dir, "UTU_logo_EN_RGB.png")  # Replace with your left icon path
        if os.path.exists(icon_left_path):
            left_icon_width = 165.6
            left_icon_height = 61.6  # Adjusted to maintain aspect ratio (828x308)
            margin = 50  # Consistent margin from the edges
            self.c.setFillAlpha(opacity)
            self.c.drawImage(icon_left_path, x=margin, y=A4[1] - left_icon_height*scale - margin-15, 
                            width=left_icon_width*scale, height=left_icon_height*scale, mask='auto')
        else:
            print(f"Warning: Left icon not found at {icon_left_path}")

        # Add the second icon (right)
        icon_right_path = os.path.join(icon_dir, "smat_logo_png.png")  # Replace with your right icon path
        if os.path.exists(icon_right_path):
            right_icon_width = 50.1
            right_icon_height = 63.83  # 301x383
            # Align the left edge of the right icon with the right margin
            self.c.drawImage(icon_right_path, x=A4[0] - right_icon_width*scale - margin, 
                            y=A4[1] - right_icon_height*scale - margin-15, 
                            width=right_icon_width*scale, height=right_icon_height*scale, mask='auto')    
        else:
            print(f"Warning: Right icon not found at {icon_right_path}")

    def _add_cover_page(self):
        """Creates the cover page with logos and title."""

        # Set starting Y position (adjust this to move entire block up/down)
        title_block_y = A4[1] * 0.6  # Starts at 60% of page height

        # Main title line 1 centered horizontally
        self.c.setFont("Helvetica-Bold", 20)
        line1 = "Photovoltaic Performance Report:"
        line1_width = self.c.stringWidth(line1, "Helvetica-Bold", 20)
        self.c.drawString((A4[0] - line1_width)/2, title_block_y, line1)

        # Main title line 2 (with slightly larger font for emphasis)
        self.c.setFont("Helvetica-Bold", 20)
        line2 = "8-Pixel Device Analysis"
        line2_width = self.c.stringWidth(line2, "Helvetica-Bold", 20)
        self.c.drawString((A4[0] - line2_width)/2, title_block_y - 30, line2)  # 30 units below line 1

        # Experimenter name (smaller font, below title)
        self.c.setFont("Helvetica", 14)
        experimenter_text = f"Experimenter: {self.experimenter}"
        exp_width = self.c.stringWidth(experimenter_text, "Helvetica", 14)
        self.c.drawString((A4[0] - exp_width)/2, title_block_y - 70, experimenter_text)  # 70 units below title start
            
        # Experiment date
        date_text = f'Report Generated: {datetime.now().strftime("%d/%m/%Y")}'
        self.c.setFont("Helvetica", 12)
        date_width = self.c.stringWidth(date_text, "Helvetica", 12)
        self.c.drawString((A4[0] - date_width) / 2, 40, date_text)

        self._add_icons()
        self.c.showPage()
    
    def _add_sheet_results(self, sample, px, jv_data_path, performance_data_path, graph_count):
        """Processes each pixel data and adds its results to the PDF."""
        # sample: sample name string
        # px: pixel number (int)
        # jv_data_path: path to JV CSV
        # performance_data_path: path to performance CSV
        # graph_count: sequential number for figures/tables

        # Parse JV and performance data for this sample/pixel
        parser = EightPixelDataAnalyzer(jv_data_path, performance_data_path, sample, px)
        metadata = parser.get_metadata()
        light_intensity = metadata.get("light_intensity", 'N/A')[0]
        page_width, page_height = A4  # Get the page dimensions

        # Page title
        self.c.setFont("Helvetica-Bold", 14)
        page_title = f"Results of {sample}: Pixel Number {px}"
        text_width = self.c.stringWidth(page_title, "Helvetica-Bold", 14)
        self.c.drawString((page_width - text_width) / 2, page_height - 65, page_title)

        # Graph size and positioning (horizontally centered)
        graph_width = 400  # Adjusted width
        graph_height = 300  # Adjusted height
        graph_x_position = (page_width - graph_width) / 2  # Horizontally center the graph
        graph_y_position = 400  # Adjusted y-position for the graph, this parameter sets the positions of the graphs and tables dynamically

        plot = parser.generate_jv_graph()
        self.c.drawImage(plot, graph_x_position, graph_y_position, width=graph_width, height=graph_height)

        # Graph title just below the graph
        graph_title = f"Figure {graph_count}. {'Dark' if light_intensity==0 else 'Light'} J-V Curve of {sample}: Pixel {px}"
        graph_title_width = self.c.stringWidth(graph_title, "Helvetica", 12)
        self.c.setFont("Helvetica", 12)
        self.c.drawString((page_width - graph_title_width) / 2, graph_y_position - 20, graph_title)

        # Add Table Title below the graph title
        table_title = f"Table {graph_count}. Performance Data of {sample}: Pixel {px}"
        table_title_width = self.c.stringWidth(table_title, "Helvetica", 12)
        self.c.setFont("Helvetica", 12)
        table_title_height=graph_y_position - 50
        self.c.drawString((page_width - table_title_width) / 2, table_title_height, table_title)

        # Add Table (Horizontally arranged)
        table_data = parser.construct_table()

        table_data_transposed = list(zip(*table_data))  # Transpose for dual-scan format

        table = Table(table_data_transposed)
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),  # Reduce font size to 8
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        table.wrapOn(self.c, page_width - 100, page_height)
        table_width = table._width  # Get the actual width of the table
        table_height = table._height  # Get actual rendered height
        
        # Dynamically adjust vertical spacing
        BUFFER_SPACE = 20  # Space between title and table (adjust as needed)
        table_y_position = table_title_height - table_height - BUFFER_SPACE
        table_height = table._height  # Get actual rendered height

        table.drawOn(self.c, (page_width - table_width) / 2 , table_y_position)  # Adjusted position for the table
        
        # Calculate and display Hysteresis Index if available
        hi_value = parser.calculate_hi()

        if hi_value is not None:
            # Displaying calculated HI
            self.c.setFont("Helvetica", 12)
            page_title = f"The calculated Hysteresis Index (HI) for Pixel {px} is {hi_value}"
            text_width = self.c.stringWidth(page_title, "Helvetica", 12)
            left_margin = 56.7
            self.c.drawString(left_margin, table_y_position - 30, page_title)

        # Title for metadata
        self.c.setFont("Helvetica-Bold", 13)
        page_title = f"Metadata of {sample} Pixel {px}:"
        text_width = self.c.stringWidth(page_title, "Helvetica-Bold", 13)
        self.c.drawString(56.7, table_y_position - 60, page_title)

        # Extract key metadata fields
        start_time = metadata.get('start_time', 'N/A')
        sample_area = metadata.get('sample_area', 'N/A')
        start_voltage = metadata.get('start_voltage', 'N/A')
        stop_voltage = metadata.get('stop_voltage', 'N/A')
        sweep_speed = metadata.get('sweep_speed', 'N/A')
        direction = metadata.get('direction', 'N/A')
        ir_temp = metadata.get('ir_temperature', 'N/A')

        # Set font for metadata block
        self.c.setFont("Helvetica", 11)

        # Starting Y position
        y_position = table_y_position - 80
        line_height = 16

        # Draw metadata
        metadata_lines = [
            f"Start Time: {start_time}",
            f"Sample Area: {sample_area} cm²",
            f"Voltage Range: {start_voltage} V to {stop_voltage} V",
            f"Sweep Speed: {sweep_speed}  mV/s",
            f"Scan Direction: {direction}",
            f"IR Temperature: {ir_temp} °C"
        ]

        for line in metadata_lines:
            self.c.drawString(56.7, y_position, line)  # 72 points = 1 inch margin
            y_position -= line_height     

        self._add_page_number()
        self._add_icons(opacity = 0.5, scale=0.6)
        self.c.showPage()
        return True
    
    def _add_box_plot_page(self, graph_count, filepaths):
        """Generates the final page with boxplots arranged in a 2x2 grid, with the title below the plots."""
        page_width, page_height = A4  # Get the page dimensions
        self.analyzed_8_pixel_sample =  EightPixelSampleAnalyzer(filepaths) # Aggregate all pixel data to generate summary boxplots

        # Page title
        self.c.setFont("Helvetica-Bold", 14)
        page_title = "Solar Cell Key Performance Metrics"
        text_width = self.c.stringWidth(page_title, "Helvetica-Bold", 14)
        self.c.drawString((page_width - text_width) / 2, page_height - 50, page_title)

        # Define the size and positions for the plots in the grid
        plot_width = 250
        plot_height = 200
        margin_x = 50
        margin_y = 200
        x_gap = 250  # Horizontal gap between plots
        y_gap = 200  # Vertical gap between plots

        # Plotting PCE Values (Top-left)
        plot = TwoPixelMeasurementAnalyzer.plot_boxplot(self.analyzed_8_pixel_sample.pce.fw, self.analyzed_8_pixel_sample.pce.rv, 'PCE values', 'PCE (%)')
        self.c.drawImage(plot, margin_x, margin_y + y_gap, width=plot_width, height=plot_height)
        self.c.setFont('Helvetica', 10)
        self.c.drawString(margin_x, margin_y + y_gap + 185, "(a)")

        # Plotting Fill Factor Values (Top-right)
        plot = TwoPixelMeasurementAnalyzer.plot_boxplot(self.analyzed_8_pixel_sample.ff.fw, self.analyzed_8_pixel_sample.ff.rv, 'Fill Factor values', 'FF')
        self.c.drawImage(plot, margin_x + x_gap, margin_y + y_gap, width=plot_width, height=plot_height)
        self.c.drawString(margin_x + x_gap , margin_y + y_gap + 185, "(b)")

        # Plotting J sc (Bottom-left)
        plot = TwoPixelMeasurementAnalyzer.plot_boxplot(self.analyzed_8_pixel_sample.isc.fw, self.analyzed_8_pixel_sample.isc.rv, 'Short Circuit Current Densities', 'Isc (mA)')
        self.c.drawImage(plot, margin_x, margin_y, width=plot_width, height=plot_height)
        self.c.drawString(margin_x, margin_y + 185, "(c)")

        # Plotting V oc (Bottom-right)
        plot = TwoPixelMeasurementAnalyzer.plot_boxplot(self.analyzed_8_pixel_sample.voc.fw, self.analyzed_8_pixel_sample.voc.rv, 'Open Circuit Voltage values', 'Voc (V)')
        self.c.drawImage(plot, margin_x + x_gap, margin_y, width=plot_width, height=plot_height)
        self.c.drawString(margin_x + x_gap, margin_y + 185, "(d)")
        
        # Add the title below the graphs
        self.c.setFont('Helvetica', 12)
        figure_title = f"Figure {graph_count}. (a) PCE, (b) FF, (c) Jsc, (d) Voc of the samples"
        title_width = self.c.stringWidth(figure_title, 'Helvetica', 12)
        self.c.drawString((page_width - title_width) / 2, margin_y - 15, figure_title)

        # Add page number and icons
        self._add_page_number()
        self._add_icons(opacity=0.5, scale=0.6)

        # Increment graph count for subsequent pages
        graph_count += 1
        self.c.showPage()

    def save_pdf(self):
        """Runs the full report generation process."""

        self._add_cover_page()

        # Count how many graphs have been generated so far (for report formatting)
        graph_count = 1

        for sample, pixels in self.jv_file_paths.items():
            performance_filepaths = [filepaths[i] for filepaths in pixels.values() for i in range(1, len(filepaths), 2)] # To store performance data

            for px, files in pixels.items():
                for file_index  in range (0, len (files),2):
                    jv_data_path = files[file_index ]
                    performance_data_path = files [file_index +1]
                    self._add_sheet_results(sample, px, jv_data_path, performance_data_path, graph_count)

            self._add_box_plot_page(graph_count, performance_filepaths)

        # Save PDF to disk  
        self.c.save()
        print(f"PDF generated: {self.pdf_path}")