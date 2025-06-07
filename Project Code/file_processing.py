import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from dataclasses import dataclass
from typing import List
from io import StringIO
import re
import math
import random

# Define the base directory
base_dir = os.getcwd()
plots_dir = os.path.join(base_dir, "Plots")

# Ensure Plots directory exists
os.makedirs(plots_dir, exist_ok=True)


class TwoPixelDataAnalyzer:
    """
    Processes J–V measurement curve data for a single sample from 2-pixel Excel sheets.

    Responsibilities:
    - Load and clean data from the specified sheet and its paired reverse/forward sheet if provided.
    - Validate that each DataFrame contains the required J–V and P–V columns with non-null values.
    - Generate J–V/P–V plots and calculate photovoltaic performance metrics (e.g., PCE, FF, Jsc, Voc, HI).
    """
    
    def __init__(self, file_path, sheet_name, next_sheet_name=None):
        self.file_path = file_path # File path of the selected excel file
        self.sheet_name = sheet_name # The selected sheet name in the excel file
        self.data = self._load_data(sheet_name) # Loading the data
        self.df_valid = self._validate_data(self.data) # Checking if the sheet contains valid data

        self.data2 = None # Setting the dataframe for the paired sheet None
        self.df2_valid = None # Setting the condition of the dataframe for the paired  sheet None

        if next_sheet_name is not None: # Checking whether the paired sheet name is provided
            self.next_sheet_name=next_sheet_name # Setting the name of th paired sheet to a variable
            self.data2 = self._load_data(next_sheet_name) # Acquiring the data from the paired sheet
            self.df2_valid =self._validate_data(self.data2) # Checking the validity of the data stored in the paired sheet

    def _load_data(self,sheet_name):
        """Load Excel data from the specified sheet."""
        try:
            df = pd.read_excel(self.file_path, sheet_name=sheet_name)
            return df
        except Exception as e:
            print(f"Error loading {sheet_name}: {e}")
            return None

    @staticmethod
    def _validate_data(df):
        """Check if the dataframe contains valid J-V measurement data."""
        # Check if at least one dataframe has valid data
        df_valid = df is not None and 'VOLTAGE (V).1' in df.columns and 'mA/cm²' in df.columns and not df[['VOLTAGE (V).1', 'mA/cm²']].isnull().all().all()
        return df_valid

    def generate_jv_graph(self):
        """Generate and save J-V and P-V curves, including the next sheet if provided."""

        if not self.df_valid and not self.df2_valid:
            return None  # Skip plotting if both dataframes are empty
        
        df= self.data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate a timestamp for the filename
        fig, ax1 = plt.subplots()  # Create figure and primary y-axis
        ax2 = ax1.twinx()  # Always create ax2 (for the secondary axis)

        if self.df_valid:            
            # Plot J-V curve
            direction = '(fwd)' if '-fw' in self.sheet_name else '(rv)'
            ax1.plot(df['VOLTAGE (V).1'], df['mA/cm²'], label= f'J-V {direction}', color='#66b3ff')
            ax1.set_xlabel("Voltage (V)", fontsize=12)  # Increased font size
            ax1.set_ylabel("Current Density (mA/cm²)", color='black', fontsize=12)  # Increased font size
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.axhline(0, color='black')  # Add a horizontal line at J = 0
            ax1.axvline(0, color='black')  # Add a horizontal line at V = 0

            # Plot P-V curve on a secondary axis
            ax2.plot(df['V'], df['mW/cm²'], label=f'P-V {direction}', color='#66b3ff', linestyle='--')  # Dashed line for P-V
            ax2.set_ylabel("Power Density (mW/cm²)", color='black', fontsize=12)  # Increased font size
            ax2.tick_params(axis='y', labelcolor='black')

            # Align the 0 values of both y-axes
            ax2.set_ylim(ax1.get_ylim())

            # Find Maximum Power Point (MPP) and plot points
            max_power_index = df['mW/cm²'].idxmax()
            mpp_voltage = df['V'].iloc[max_power_index]
            mpp_current_density = df['mA/cm²'].iloc[max_power_index]
            mpp_power_density = df['mW/cm²'].iloc[max_power_index]

            # Plot red points at MPP
            ax1.plot(mpp_voltage, mpp_current_density, 'o', color='#66b3ff', label=r"$J_{\mathit{mp}}$ " + ('(fwd)' if '-fw' in self.sheet_name else '(rv)'))  # Red point for J-V curve
            ax2.plot(mpp_voltage, mpp_power_density, 'o', color='#66b3ff', label=r"$P_{\mathit{mp}}$" + ('(fwd)' if '-fw' in self.sheet_name else '(rv)'))  # Red point for P-V curve

        # Plot df2 if it exists and has required columns
        if self.df2_valid:

            df2 = self.data2
            direction = '(fwd)' if '-fw' in self.next_sheet_name else '(rv)'
            # Plot J-V curve for df2
            ax1.plot(df2['VOLTAGE (V).1'], df2['mA/cm²'], label=f'J-V {direction}', color='darkblue')

            # Plot P-V curve for df2
            ax2.plot(df2['V'], df2['mW/cm²'], label=f'P-V {direction}', color='darkblue', linestyle='--')

            # Find and plot MPP for df2
            max_power_index_2 = df2['mW/cm²'].idxmax()
            mpp_voltage_2 = df2['V'].iloc[max_power_index_2]
            mpp_current_density_2 = df2['mA/cm²'].iloc[max_power_index_2]
            mpp_power_density_2 = df2['mW/cm²'].iloc[max_power_index_2]

            # Plot blue points at MPP
            ax1.plot(mpp_voltage_2, mpp_current_density_2, 'bo', label=r"$J_{\mathit{mp}}$" + direction)
            ax2.plot(mpp_voltage_2, mpp_power_density_2, 'bo', label=r"$P_{\mathit{mp}}$" +  direction)

        # Align the 0 values of both y-axes
        ax2.set_ylim(ax1.get_ylim())

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        total_labels = labels1 + labels2
        num_columns = (len(total_labels) + 1) // 2  # Divide into two rows
        ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.2),  # Adjust legend position
        fontsize=9,  # Smaller font size for legend
        borderpad=0.7,  # Reduce padding inside the legend box
        frameon=False,  # Remove legend border
        ncol=num_columns  # Split into two rows
        )

        # Save the plot
        plot_filename = f"j_v_plot_{self.sheet_name}_{timestamp}.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Saved plot for {self.sheet_name} and {self.next_sheet_name}: {plot_path}")
        return plot_path
    
    def calculate_hi(self):
        """Calculate the Hysteresis Index (HI) if both forward and reverse scan PCE values are available."""
        
        if not self.df_valid or not self.df2_valid:
            return None  # Skip if  dataframes are empty
        
        # Compute PCE values directly
        pce = round(self.data['mW/cm²'].max(), 3) 
        pce2 = round(self.data2['mW/cm²'].max(), 3) 

        # Determine which scan is reverse and which is forward
        scan_type_1 = 'Forward Scan' if '-fw' in self.sheet_name else 'Reverse Scan'

        if scan_type_1 == "Reverse Scan":
            pce_reverse, pce_forward = pce, pce2
        else:
            pce_reverse, pce_forward = pce2, pce

        # Compute HI
        hi_value = round((pce_reverse - pce_forward) / pce_reverse, 3)
        
        return hi_value


    @staticmethod
    def extract_performance_data(df):
        df = df.dropna(subset=['VOLTAGE (V).1', 'mA/cm²']).copy()

        """Extract common photovoltaic parameters from a given dataframe."""
        active_area = df.iloc[2, 13] # Extracting active area of the cell
        pce = round(df['mW/cm²'].max(), 3) # Extracting PCE of the cell

        max_row = df.loc[df['mW/cm²'].idxmax()]
        max_voltage = round(max_row['VOLTAGE (V).1'], 3) # Extracting voltage at maximum power
        max_current = round(max_row['mA/cm²'], 3) # Extracting current density at maximum power

        # Estimate J_sc (Short Circuit Current Density)
        zero_voltage_row = df[df['VOLTAGE (V).1'] == 0]
        if not zero_voltage_row.empty:
            mA_at_zero_voltage = round(zero_voltage_row['mA/cm²'].values[0], 3)
        else: # If no row has a voltage of exactly zero, the code estimates the current density at zero voltage using linear regression
            closest_indices = np.argsort(np.abs(df['VOLTAGE (V).1'].values))[:2]
            slope, intercept, *_ = stats.linregress(df.iloc[closest_indices]['VOLTAGE (V).1'].values,
                                                    df.iloc[closest_indices]['mA/cm²'].values)
            mA_at_zero_voltage = round(intercept, 3)

        # Estimate V_oc (Open Circuit Voltage)
        zero_current_row = df[df['mA/cm²'] == 0]
        if not zero_current_row.empty:
            V_at_zero_current = round(zero_current_row['VOLTAGE (V).1'].values[0], 3)
        else: # If no row has a current density of exactly zero, the code estimates the current density at zero voltage using linear regression
            closest_indices = np.argsort(np.abs(df['mA/cm²'].values))[:2]
            slope, intercept, *_ = stats.linregress(df.iloc[closest_indices]['mA/cm²'].values,
                                                    df.iloc[closest_indices]['VOLTAGE (V).1'].values)
            V_at_zero_current = round(intercept, 3)

        fill_factor = round((max_voltage * max_current) / (V_at_zero_current * mA_at_zero_voltage), 3)
        
        # Calculate Shunt Resistance (Rsh)
        # Using the slope near Jsc (V=0)
        near_jsc = df[(df['VOLTAGE (V).1'] >= -0.02) & (df['VOLTAGE (V).1'] <= 0.02)].copy() # Small voltage window around 0V
        near_jsc_avg = near_jsc.groupby('VOLTAGE (V).1')['mA/cm²'].mean().reset_index() # averaging repeated voltages before regression to reduce noise.
        epsilon = 1e-6  # tiny threshold to avoid div by zero
        R_sh = None

        if len(near_jsc_avg) > 1:
            # Convert J from mA/cm² to A/cm²
            near_jsc_avg.loc[:, 'J_A_cm2'] = near_jsc_avg['mA/cm²'] * 1e-3

            slope, *_ = stats.linregress(near_jsc_avg['VOLTAGE (V).1'], near_jsc_avg['J_A_cm2'])
          
            if abs(slope) > epsilon:
                R_sh = round(1/abs(slope), 3)  # Rsh = dV/dJ (ohm*cm²)

        # Calculate Series Resistance (Rs)
        # Using the slope near Voc (J=0)
        near_voc = df[(df['mA/cm²'] >= -1) & (df['mA/cm²'] <= 1)].copy()  # Small current window around 0mA
        near_voc_avg = near_voc.groupby('mA/cm²')['VOLTAGE (V).1'].mean().reset_index() # averaging repeated currents before regression to reduce noise.
        R_s = None

        if len(near_voc_avg) > 1:
            # Convert J from mA/cm² to A/cm²
            near_voc_avg.loc[:, 'J_A_cm2'] = near_voc_avg['mA/cm²'] * 1e-3

            slope, *_ = stats.linregress(near_voc_avg['J_A_cm2'], near_voc_avg['VOLTAGE (V).1'])
            
            if abs(slope) > epsilon:
                R_s = round(abs(slope), 3)  # Rs = dV/dJ (ohm*cm²)

        return active_area, pce, max_voltage, max_current, mA_at_zero_voltage, V_at_zero_current, fill_factor, R_sh, R_s
    
    def construct_table(self):
        """Constructing a table from key photovoltaic parameters in the data."""
        # Initialize table headers
        table = [["Parameter", "Units"]]

        # Check if the first dataset is valid
        if self.df_valid:
            table[0].append("Forward Scan" if "-fw" in self.sheet_name else "Reverse Scan")
            active_area, pce, max_voltage, max_current, mA_at_zero_voltage, V_at_zero_current, fill_factor, R_sh, R_s = TwoPixelDataAnalyzer.extract_performance_data(self.data)
        
        # Check if the second dataset is valid
        if self.df2_valid:
            table[0].append("Forward Scan" if "-fw" in self.next_sheet_name else "Reverse Scan")
            active_area2, pce2, max_voltage2, max_current2, mA_at_zero_voltage2, V_at_zero_current2, fill_factor2, R_sh2, R_s2 = TwoPixelDataAnalyzer.extract_performance_data(self.data2)
       
        # Populate the table dynamically
        params = [
            ("Active Area","cm²", active_area if self.df_valid else None, active_area2 if self.df2_valid else None),
            ("Jsc","mA/cm²", mA_at_zero_voltage if self.df_valid else None, mA_at_zero_voltage2 if self.df2_valid else None),
            ("Voc","V", V_at_zero_current if self.df_valid else None, V_at_zero_current2 if self.df2_valid else None),
            ("Fill Factor","-", fill_factor if self.df_valid else None, fill_factor2 if self.df2_valid else None),
            ("PCE", "%", pce if self.df_valid else None, pce2 if self.df2_valid else None),
            ("Jmp", "mA/cm²", max_current if self.df_valid else None, max_current2 if self.df2_valid else None),
            ("Vmp", "V", max_voltage if self.df_valid else None, max_voltage2 if self.df2_valid else None),
            ("Rsh", "Ω·cm²", R_sh if self.df_valid else None, R_sh2 if self.df2_valid else None),
            ("Rs", "Ω·cm²", R_s if self.df_valid else None, R_s2 if self.df2_valid else None),
        ]

        for param in params:
            row = [param[0],param[1]]
            if self.df_valid:
                row.append(param[2])
            if self.df2_valid:
                row.append(param[3])
            table.append(row)
        return table
    
@dataclass
class MeasurementResults:
    """
    Data container for photovoltaic measurement results from both scan directions.

    Attributes:
        fw (List[float]): Numeric results collected during the forward scan.
        rv (List[float]): Numeric results collected during the reverse scan.
        all (List[float]): Combined list of forward and reverse scan values for aggregate analysis.
    """

    fw: List[float]
    rv: List[float]
    all: List[float]


class TwoPixelMeasurementAnalyzer:
    """
    Parses and summarizes photovoltaic measurement data from a 2-pixel Excel file.

    Responsibilities:
    - Load the Excel workbook and identify all measurement sheets.
    - For each sheet, compute key solar cell parameters (Jsc, Voc, FF, PCE).
    - Segregate forward and reverse scan data and aggregate combined results.
    - Provide processed data for boxplot generation.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path # The file path of the selected file
        self.xls = pd.ExcelFile(self.file_path) # Reading the selected file
        self.sheets = self.xls.sheet_names # Acquiring the sheet names from the selected file
        
        # Initialize structured results storage
        self.jsc = MeasurementResults([], [], [])  # Short-circuit current density
        self.voc = MeasurementResults([], [], [])  # Open-circuit voltage
        self.ff = MeasurementResults([], [], [])   # Fill factor
        self.pce = MeasurementResults([], [], [])  # Power conversion efficiency
        
        # Process all sheets
        self._analyze_sheets()
    
    def _analyze_sheets(self):
        """Process all sheets in the Excel file."""

        for sheet in self.sheets:
            df = pd.read_excel(self.xls, sheet_name=sheet)
            
            if not self._has_required_columns(df):
                continue
                
            # Extract key parameters
            jsc = self._extract_jsc(df)
            voc = self._extract_voc(df)
            
            # Store results based on scan direction
            if '-fw' in sheet:
                self.jsc.fw.append(jsc)
                self.voc.fw.append(voc)
            else:
                self.jsc.rv.append(jsc)
                self.voc.rv.append(voc)
                
            self.jsc.all.append(jsc)
            self.voc.all.append(voc)
            
            # Calculate derived parameters
            self._calculate_derived_parameters(df, sheet, jsc, voc)

    def _has_required_columns(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame contains required columns."""
        required = ['VOLTAGE (V).1', 'mA/cm²']
        return all(col in df.columns for col in required) and not df[required].isnull().all().all()
    
    
    def _extract_jsc(self, df: pd.DataFrame) -> float:
        """Extract short-circuit current density (Jsc)."""
        return self._extrapolate_zero_x(df, 'VOLTAGE (V).1', 'mA/cm²')
    
    def _extract_voc(self, df: pd.DataFrame) -> float:
        """Extract open-circuit voltage (Voc)."""
        return self._extrapolate_zero_x(df, 'mA/cm²', 'VOLTAGE (V).1')
    
    def _extrapolate_zero_x(self, df: pd.DataFrame, x_col: str, y_col: str) -> float:
        """
        Generic method to find y-value where x=0 using either direct measurement or linear extrapolation.
        """
        zero_x_row = df[df[x_col] == 0]
        if not zero_x_row.empty:
            return round(zero_x_row[y_col].values[0], 3)
        
        # Linear extrapolation if no exact zero point
        closest_indices = np.argsort(np.abs(df[x_col].values))[:2]
        slope, intercept, *_ = stats.linregress(
            df.iloc[closest_indices][x_col].values,
            df.iloc[closest_indices][y_col].values
        )
        return round(intercept, 3)
    
    def _calculate_derived_parameters(self, df, sheet, jsc, voc):
        """Calculate fill factor and PCE."""
        max_row = df.loc[df['mW/cm²'].idxmax()]
        max_voltage = round(max_row['VOLTAGE (V).1'], 3)
        max_current = round(max_row['mA/cm²'], 3)
        
        ff = round((max_voltage * max_current) / (voc * jsc), 3)
        pce = round(df['mW/cm²'].max(), 3)
        
        if '-fw' in sheet:
            self.ff.fw.append(ff)
            self.pce.fw.append(pce)
        else:
            self.ff.rv.append(ff)
            self.pce.rv.append(pce)
            
        self.ff.all.append(ff)
        self.pce.all.append(pce)

    @staticmethod
    def plot_boxplot(fw_values, rv_values,
                title_text, y_text):
        """Generate and save a boxplot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.figure(figsize=(6, 4))

        # Create a black-and-white boxplot
        all_values = fw_values + rv_values

        sns.boxplot(data=all_values, 
            color='white',  # White fill
            linewidth=1.5,  # Thicker black borders
            boxprops=dict(edgecolor='black'),  # Box border color
            whiskerprops=dict(color='black'),  # Whisker color
            capprops=dict(color='black'),  # Cap line color
            medianprops=dict(color='black'),  # Median line color
            flierprops=dict(markerfacecolor='none', markeredgecolor='black', markersize=6, marker = 'o'))  # Outliers
        
        # Calculate the IQR to identify outliers
        Q1 = np.percentile(all_values, 25)
        Q3 = np.percentile(all_values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out the outliers for each list based on the combined thresholds
        filtered_fw_values = [x for x in fw_values if lower_bound <= x <= upper_bound]
        filtered_rv_values = [x for x in rv_values if lower_bound <= x <= upper_bound]
        
        # Overlay the data points
        sns.stripplot(data=filtered_fw_values, color='#66b3ff', jitter=True, size=5, label="Forward Scan", dodge=True)
        sns.stripplot(data=filtered_rv_values, color='darkblue', jitter=True, size=5, label="Reverse Scan", dodge=True)
   
        plt.title(f'Boxplot of {title_text}', fontsize =16)
        plt.ylabel(y_text,labelpad=15, fontsize=12)

        # Increase the font size of the tick labels
        plt.yticks(fontsize=16)

        # Show legend in the bottom-right corner
        plt.legend(loc='lower right',
                   frameon=False)

        # Adjust layout to prevent clipping
        plt.tight_layout()

        plot_filename = f"boxplot_of_{title_text}_{timestamp}.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved PCE boxplot: {plot_path}")
        return plot_path
    
class EightPixelDataAnalyzer:

    """
    Parses and analyzes J–V and performance CSV files for a single 8-pixel sample.

    Responsibilities:
    - Read raw J–V data file, clean lines, extract metadata, and split bidirectional scans.
    - Validate and transform voltage/current data into pandas DataFrames.
    - Read performance CSV, parse MPP, Voc, Jsc, FF, PCE, and compute series/shunt resistances.
    - Provide accessors for metadata, raw JV DataFrames, and generate J–V/P–V plots.
    """

    def __init__(self, jv_data_path, performance_data_path, sample, px):
        self.sample = sample # The Sample name
        self.px = px # The Pixel Number
        self.jv_data_path = jv_data_path # The file path of the J_V csv file
        self.performance_data_path = performance_data_path # The file path of the performance data csv file

        # Placeholder for all extracted metadata fields
        self.metadata = {
            "start_time": None,
            "sample_area": None,
            "start_voltage": None,
            "stop_voltage": None,
            "sweep_speed": None,
            "direction": None,
            "ir_temperature": None,
            "light_intensity": None
        }
        
        # DataFrames for split scans
        self.df_1 = None
        self.df_2 = None
        self._parse_file() # Parse header & data sections from the raw JV file
        self._validate_data() # Validate that df_1/df_2 contain actual data
        self._modify_data() # Convert raw currents to densities and filter
        self.get_performance_data() # Load performance CSV and extract PV parameters
    
    @staticmethod
    def _clean_line(line):
        """
        Remove unwanted characters and trailing whitespace from a line.
        Handles encoding issues and formatting inconsistencies.
        """
        # Normalize non-breaking spaces and tabs for consistent parsing
        return line.replace('\xa0', ' ').replace('\t', ' ').strip()

    def _parse_file(self):
        """
        Read the file and trigger metadata and data extraction.
        Handles encoding issues during file reading.
        """
        try:
            with open(self.jv_data_path, encoding='latin1') as f:
                lines = [self._clean_line(line) for line in f]
        except Exception as e:
            # If file can’t be read (encoding issue, missing file), abort parsing
            print(f"❌ Failed to read file {self.jv_data_path}: {e}")
            return

        self._extract_metadata(lines)
        self._extract_data(lines)

    def _extract_metadata(self, lines):
        """
        Extract metadata fields from header lines.
        Each metadata value is identified using specific string patterns and regex.
        """
        # Scan every header line for known metadata markers
        for line in lines:
            if "#start_time:" in line: # Extract the start time
                self.metadata["start_time"] = line.split(":", 1)[1].strip()
            elif "#Sample area:" in line: # Extract the sample area
                match = re.search(r"([\d.]+)", line)
                self.metadata["sample_area"] = float(match.group(1)) if match else None
            elif "#start_voltage:" in line: # Extract the starting voltage
                match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", line)
                self.metadata["start_voltage"] = float(match.group(1)) if match else None
            elif "#stop_voltage:" in line: # Extract the stop voltage
                match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", line)
                self.metadata["stop_voltage"] = float(match.group(1)) if match else None
            elif "#sweep_speed:" in line or "#Scan rate:" in line: # Extract the sweep speed
                match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", line)
                self.metadata["sweep_speed"] = float(match.group(1)) if match else None
            elif "#direction:" in line:
                # Extract the direction number (0, 1, 2, or 3)
                direction_num = int(re.search(r"(\d+)", line).group(1))
                
                # Map the number to its description
                direction_map = {
                    0: "Forward Scan",
                    1: "Backward Scan",
                    2: "Forward then Backward Scan",
                    3: "Backward then Forward Scan"
                }
                
                # Store both the number and description
                self.metadata["direction"] = direction_map.get(direction_num, None)
                self.direction = self.metadata["direction"]  # Description (string)
                self.direction_num = direction_num           # Number (0, 1, 2, or 3)
            elif "##IR temperature [0]:" in line: # Extract the Infra Red temperature
                match = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
                self.metadata["ir_temperature"] = [float(x) for x in match[1:]]
            elif "##Light intensity" in line: # Extract the light intensity
                match = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
                self.metadata["light_intensity"] = [float(x) for x in match]

    def _extract_data(self, lines):
        """
        Parse the tabular voltage-current data from the file body.
        Uses the header keywords to locate the start of the data block,
        then parses into a DataFrame using flexible delimiters.
        """

        # Find the first line containing both ‘voltage’ and ‘current’ headers
        header_keywords = ['voltage', 'current']

        data_start_index = next(
            (i + 1 for i, line in enumerate(lines) if all(k in line.lower() for k in header_keywords)),
            None
        )

        if data_start_index is None:
            print(f"❌ Data section not found in {self.jv_data_path}")
            return

        data_block = "\n".join(lines[data_start_index:])

        try:
            # For bidirectional scans, split the DF at NaN row
            df = pd.read_csv(StringIO(data_block), sep=r'\s+|,', engine='python', names=['Voltage (V)', 'Current (mA)'])

            if self.direction_num in [2,3]:
                # Split at the first Nan in Voltage or Current
                nan_idx= df[df.isna().any(axis=1)].index

                if not nan_idx.empty:
                    split_idx = nan_idx[0]
                    self.df_1 = df.iloc[:split_idx].dropna()
                    self.df_2 = df.iloc[split_idx+1:].dropna()
                else:
                    # Warn if no NaN marker (unexpected format) or parsing fails
                    print(f"⚠️ No NaN marker found for direction {self.direction_num} in {self.jv_data_path}")
                    self.df = df.dropna()
            else:
                self.df_1 = df.dropna()

        except Exception as e:
            print(f"❌ DataFrame parsing failed in {self.jv_data_path}: {e}")
            self.df = pd.DataFrame()

    def get_metadata(self):
        """
        Return the extracted metadata dictionary.
        """
        return self.metadata

    def get_dataframe(self):
        """
        Return the parsed voltage-current data as a DataFrame.
        """
        return self.df_1, self.df_2
    
    def get_performance_data(self):
        """
        Extracts V_mpp, I_mpp, MPP, Voc, Jsc, FF, PCE, Rs, and Rsh from a performance data CSV file.
        """

        # Read the CSV file
        df = pd.read_csv(self.performance_data_path, encoding='latin1', sep=';')

        # Locate the '#Vmpp' marker row and parse comma-separated values
        vmpp_index = df[df.iloc[:, 0].str.contains('#Vmpp', na=False)].index[0]

        def parse_parameters(row_offset):
            row = df.iloc[vmpp_index + row_offset, 0]
            params = [p.strip() for p in row.split(',')]
            return params

        def assign_metrics(prefix, params):
            # assign_metrics: map parsed values to attributes (mpp_voltage, jsc, pce, etc.)
            sample_area = self.metadata["sample_area"]
            setattr(self, f'{prefix}_mpp_voltage', round(float(params[0]), 3))
            setattr(self, f'{prefix}_mpp_current_density', round(float(params[1]) / sample_area * -1, 3))
            setattr(self, f'{prefix}_mpp', round(float(params[2]) / sample_area * -1, 3))
            setattr(self, f'{prefix}_voc', round(float(params[3]), 3))
            setattr(self, f'{prefix}_jsc', round(float(params[4]) / sample_area * -1, 3))
            setattr(self, f'{prefix}_ff', round(float(params[5]), 3))
            setattr(self, f'{prefix}_hi', round(float(params[6]), 3))
            setattr(self, f'{prefix}_pce', round(float(params[7]) * 100, 3))

        def calculate_resistances(df):
            """
            Calculates and returns (Rsh, Rs) based on JV curve data.
            """
            epsilon = 1e-6
            R_sh, R_s = None, None

            # Shunt Resistance near Voc=0V
            near_jsc = df[(df['Voltage (V)'].between(-0.1, 0.1))].copy()
            near_jsc_avg = near_jsc.groupby('Voltage (V)')['Current density (mA/cm²)'].mean().reset_index()

            if len(near_jsc_avg) > 1:
                near_jsc_avg['J_A_cm2'] = near_jsc_avg['Current density (mA/cm²)'] * 1e-3
                slope, *_ = stats.linregress(near_jsc_avg['Voltage (V)'], near_jsc_avg['J_A_cm2'])
                if abs(slope) > epsilon:
                    R_sh = round(1 / abs(slope), 3)
                else:
                    print("[WARNING] Slope near Voc=0V is too small (|slope| <= epsilon). Cannot calculate R_shunt.")
            else:
                print(f"[WARNING] Not enough points between +/-0.1 to calculate R_shunt (found {len(near_jsc_avg)} points).")


            # Series Resistance near J=0 mA/cm²
            near_voc = df[(df['Current density (mA/cm²)'].between(-1, 1))].copy()
            near_voc_avg = near_voc.groupby('Current density (mA/cm²)')['Voltage (V)'].mean().reset_index()
            if len(near_voc_avg) > 1:
                near_voc_avg['J_A_cm2'] = near_voc_avg['Current density (mA/cm²)'] * 1e-3
                slope, *_ = stats.linregress(near_voc_avg['J_A_cm2'], near_voc_avg['Voltage (V)'])
                if abs(slope) > epsilon:
                    R_s = round(abs(slope), 3)
                else:
                    print("[WARNING] Slope near J=0 mA/cm² is too small (|slope| <= epsilon). Cannot calculate R_series.")
            else:
                print(f"[WARNING] Not enough points between +/-1 mA/cm² to calculate R_series (found {len(near_voc_avg)} points).")

            return R_sh, R_s

        # Extracting parameters, if there is valid data in the first scan
        if self.df_1_valid:
            df_1_params = parse_parameters(1)
            assign_metrics('df_1', df_1_params)
            self.df_1_rsh, self.df_1_rs = calculate_resistances(self.df_1)

        # Extracting parameters, if there is a second scan
        if self.df_2_valid:
            df_2_params = parse_parameters(2)
            assign_metrics('df_2', df_2_params)
            self.df_2_rsh, self.df_2_rs = calculate_resistances(self.df_2)

    def generate_jv_graph(self):
        """Generate and save J-V and P-V curves, including the next sheet if provided."""

        if self.df_1 is None and self.df_2 is None:
            return None  # Skip plotting if both dataframes are empty
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate a timestamp for the filename
        fig, ax1 = plt.subplots()  # Create figure and primary y-axis
        ax2 = ax1.twinx()  # Always create ax2 (for the secondary axis)

        # Plot df1 if it exists and has required columns
        if self.df_1_valid:
            # Decide on the color based on the direction
            color_1 = '#66b3ff' if 'Forward' in self.direction[:7] else 'darkblue'
            # Plot J-V curve
            ax1.plot(self.df_1['Voltage (V)'], self.df_1['Current density (mA/cm²)'], label= f'J-V {'(fwd)' if 'Forward' in self.direction[:7] else '(rv)'}', color=color_1)
            ax1.set_xlabel("Voltage (V)", fontsize=12)  # Increased font size
            ax1.set_ylabel("Current Density (mA/cm²)", color='black', fontsize=12)  # Increased font size
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.axhline(0, color='black')  # Add a horizontal line at J = 0
            ax1.axvline(0, color='black')  # Add a horizontal line at V = 0

            # Plot P-V curve on a secondary axis
            ax2.plot(self.df_1['Voltage (V)'], self.df_1['Current density (mA/cm²)'] * self.df_1['Voltage (V)'], label=f'P-V {'(fwd)' if 'Forward' in self.direction[:7] else '(rv)'}', color=color_1, linestyle='--')  # Dashed line for P-V
            ax2.set_ylabel("Power Density (mW/cm²)", color='black', fontsize=12)  # Increased font size
            ax2.tick_params(axis='y', labelcolor='black')

            # Align the 0 values of both y-axes
            ax2.set_ylim(ax1.get_ylim())

            # Find Maximum Power Point (MPP) and plot points
            mpp_voltage = self.df_1_mpp_voltage
            mpp_current_density = self.df_1_mpp_current_density
            mpp_power_density = self.df_1_mpp

            # Plot red points at MPP
            ax1.plot(mpp_voltage, mpp_current_density, 'o', color=color_1, label=r"$J_{\mathit{mp}}$ " + ('(fwd)' if 'Forward' in self.direction[:7] else '(rv)'))  # Red point for J-V curve
            ax2.plot(mpp_voltage, mpp_power_density, 'o', color=color_1, label=r"$P_{\mathit{mp}}$" + ('(fwd)' if 'Forward' in self.direction[:7] else '(rv)'))  # Red point for P-V curve

        # Plot df2 if it exists and has required columns
        if self.df_2_valid:
            # Decide on the color based on the direction
            color_2 = 'darkblue' if 'Forward' in self.direction[:7] else '#66b3ff'
            # Plot J-V curve for df2
            ax1.plot(self.df_2['Voltage (V)'], self.df_2['Current density (mA/cm²)'], label=f'J-V {'(rv)' if 'Forward' in self.direction[:7] else '(fwd)'}', color= color_2)

            # Plot P-V curve for df2
            ax2.plot(self.df_2['Voltage (V)'], self.df_2['Current density (mA/cm²)'] * self.df_2['Voltage (V)'], label=f'P-V {'(rv)' if 'Forward' in self.direction[:7] else '(fwd)'}', color = color_2, linestyle='--')

            # Find and plot MPP for df2
            mpp_voltage_2 = self.df_2_mpp_voltage
            mpp_current_density_2 = self.df_2_mpp_current_density
            mpp_power_density_2 = self.df_2_mpp

            # Plot blue points at MPP
            ax1.plot(mpp_voltage_2, mpp_current_density_2, 'o', color = color_2, label=r"$J_{\mathit{mp}}$" + ('(rv)' if 'Forward' in self.direction[:7] else '(fwd)'))
            ax2.plot(mpp_voltage_2, mpp_power_density_2, 'o', color = color_2, label=r"$P_{\mathit{mp}}$" +  ('(rv)' if 'Forward' in self.direction[:7] else '(fwd)'))

        # Align the 0 values of both y-axes
        ax2.set_ylim(ax1.get_ylim())

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        total_labels = labels1 + labels2
        num_columns = (len(total_labels) + 1) // 2  # Divide into two rows
        ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.2),  # Adjust legend position
        fontsize=9,  # Smaller font size for legend
        borderpad=0.7,  # Reduce padding inside the legend box
        frameon=False,  # Remove legend border
        ncol=num_columns  # Split into two rows
        )

        # Generate a random number
        random_number = random.randint(1000, 9999)

        # Save the plot
        plot_filename = f"j_v_plot_Pixel_{self.sample}_{self.px}_{timestamp}_{random_number}.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Saved plot for {self.sample} Pixel {self.px}: {plot_path}")
        return plot_path
    
    def construct_table(self):
        """Constructing a table from key photovoltaic parameters in the data."""

        # Initialize table headers
        table = [["Parameter", "Units"]]

        # Check if the first dataset is valid
        if self.df_1_valid:
            table[0].append("Forward Scan" if 'Forward' in self.direction[:7] else 'Reverse Scan')
        
        # Check if the second dataset is valid
        if self.df_2_valid:
            table[0].append("Reverse Scan" if 'Forward' in self.direction[:7] else "Forward Scan")
       
        # Populate the table dynamically
        params = [
            ("Active Area","cm²", self.metadata.get("sample_area") if self.df_1_valid else None, self.metadata.get("sample_area") if self.df_2_valid else None),
            ("Jsc","mA/cm²", self.df_1_jsc if self.df_1_valid else None, self.df_2_jsc if self.df_2_valid else None),
            ("Voc","V", self.df_1_voc if self.df_1_valid else None, self.df_2_voc if self.df_2_valid else None),
            ("Fill Factor","%", self.df_1_ff if self.df_1_valid else None, self.df_2_ff if self.df_2_valid else None),
            ("PCE", "%", self.df_1_pce if self.df_1_valid else None, self.df_2_pce if self.df_2_valid else None),
            ("Jmp", "mA/cm²", self.df_1_mpp_current_density if self.df_1_valid else None, self.df_2_mpp_current_density if self.df_2_valid else None),
            ("Vmp", "V", self.df_1_mpp_voltage if self.df_1_valid else None, self.df_2_mpp_voltage if self.df_2_valid else None),
            ("Rsh", "Ω·cm²", self.df_1_rsh if self.df_1_valid else None, self.df_2_rsh if self.df_2_valid else None),
            ("Rs", "Ω·cm²", self.df_1_rs if self.df_1_valid else None, self.df_2_rs if self.df_2_valid else None),
        ]

        for param in params:
            row = [param[0],param[1]]
            if self.df_1_valid:
                row.append(param[2])
            if self.df_2_valid:
                row.append(param[3])
            table.append(row)
        return table
    
    def calculate_hi(self):
        """Calculate the Hysteresis Index (HI) if both forward and reverse scan PCE values are available."""
        
        if self.df_1 is None or self.df_2 is None:
            return None
        
        return self.df_1_hi
    
    def _validate_data(self):
        """Check if the dataframe contains valid J-V measurement data."""
        self.df_1_valid = self.df_1 is not None and not self.df_1.empty
        self.df_2_valid = self.df_2 is not None and not self.df_2.empty
    
    def _modify_data(self):
        """
        Modify the dataframes by converting current (mA) to current density (mA/cm²),
        inverting the sign, and filtering out low current density values.
        """

        def process_df(df):
            """
            Internal helper function to process a single DataFrame:
            1. Converts 'Current (mA)' to 'Current density (mA/cm²)' using sample area.
            2. Multiplies the result by -1 to align with convention.
            3. Filters out rows with current density < -5 mA/cm².
            """
            # Retrieve the sample area from metadata
            sample_area = self.metadata.get("sample_area")

            if not sample_area:
                raise ValueError("Sample area is not defined or is zero in metadata.")

            # Convert current to current density
            df['Current density (mA/cm²)'] = df['Current (mA)'] / sample_area

            # Invert the sign of current density
            df['Current density (mA/cm²)'] *= -1

            # Filter out values with current density less than -5 mA/cm²
            df = df[df['Current density (mA/cm²)'] >= -5]

            return df

        # Apply the transformation to df_1 if valid
        if self.df_1_valid:
            self.df_1 = process_df(self.df_1)

        # Apply the transformation to df_2 if valid
        if self.df_2_valid:
            self.df_2 = process_df(self.df_2)

class EightPixelSampleAnalyzer:
    """
    Aggregates and analyzes performance metrics across all pixels of an 8-pixel sample.

    Responsibilities:
    - Classify performance CSVs into light and dark measurement sets based on header metadata.
    - Parse each CSV to extract key photovoltaic parameters (Jsc, Voc, FF, PCE).
    - Store forward, reverse, and combined metric lists in MeasurementResults containers.
    - Provide aggregated data for boxplots.
    """
    
    def __init__(self, file_path_list):
        self.file_path_list = file_path_list # List of all JV/performance CSV file paths for this sample
        
        # Initialize containers for forward, reverse, and combined metrics
        self.isc = MeasurementResults([], [], [])  # Short-circuit current density
        self.voc = MeasurementResults([], [], [])  # Open-circuit voltage
        self.ff = MeasurementResults([], [], [])   # Fill factor
        self.pce = MeasurementResults([], [], [])  # Power conversion efficiency
        self.classify_performance_files_by_light_intensity() # Split files into dark vs. light based on metadata inside each CSV
        self._analyze_sample() # Extract Jsc, Voc, FF, PCE from each classified file
    
    def _analyze_sample(self):
        """Extract the performance data from all provided performance csv files."""
        
        def extract_parameters(row):
            # Parse a comma-separated metadata line into individual fields
            return [param.strip() for param in row.split(',')]

        def store_values(direction, params, sample_area):
            # Map parsed parameters into MeasurementResults by scan direction
            jsc_val = round(-float(params[4])/ sample_area, 3)
            voc_val = round(float(params[3]), 3)
            ff_val  = round(float(params[5]), 3)
            pce_val = round(float(params[7]), 3)

            if direction in [0, 2]:
                self.isc.fw.append(jsc_val)
                self.voc.fw.append(voc_val)
                self.ff.fw.append(ff_val)
                self.pce.fw.append(pce_val)
            else:
                self.isc.rv.append(jsc_val)
                self.voc.rv.append(voc_val)
                self.ff.rv.append(ff_val)
                self.pce.rv.append(pce_val)

            self.isc.all.append(jsc_val)
            self.voc.all.append(voc_val)
            self.ff.all.append(ff_val)
            self.pce.all.append(pce_val)

        for file_path in self.light_files:
            df = pd.read_csv(file_path, encoding='latin1', sep=';')

            # Locate the row immediately following '#Vmpp' marker for performance values
            row_number = df[df.iloc[:, 0].str.contains('#Vmpp', na=False)].index[0]
            params = extract_parameters(df.iloc[row_number + 1, 0])

            # Extract scan direction (0–3) from the '#direction:' metadata line
            direction_text = df[df.iloc[:, 0].str.contains(r'#direction:', na=False)].iloc[0, 0]
            match = re.search(r'#direction:\t(\d+)', direction_text)
            direction_value = int(match.group(1)) if match else None

            if direction_value is None: # Skip if direction metadata could not be parsed
                continue
            
            # Extract sample area from the '#Sample area:' metadata line
            sample_area_text = df[df.iloc[:, 0].str.contains(r'#Sample area:', na=False)].iloc[0, 0]
            match = re.search(r'#Sample area:\t([\d\.]+)', sample_area_text)
            sample_area = float(match.group(1)) if match else None

            store_values(direction_value, params, sample_area)

            # Handle additional row (if present)
            if row_number + 2 < len(df):
                params2 = extract_parameters(df.iloc[row_number + 2, 0])
                if direction_value == 3:
                    store_values(0, params2, sample_area)  # treat second as forward
                else:
                    store_values(1, params2, sample_area)  # treat second as reverse
    
    def classify_performance_files_by_light_intensity(self):
        ''' Extracting the light intensity from the performance data csv and
            classifying the csv files into dark and light measurements
        '''
        # Prepare separate lists for light- and dark-measurement files
        self.light_files = []
        self.dark_files = []

        for path in self.file_path_list:
            try:
                # Read metadata row to determine light intensity setting
                df = pd.read_csv(path, encoding='latin1', sep=';')
                light_row = df[df.iloc[:, 0].str.contains('Light intensity', na=False)]
                if light_row.empty:
                    continue
                light_text = light_row.iloc[0, 0]
                match = re.search(r'\[(\d+(?:\.\d+)?)', light_text)
                light_intensity = float(match.group(1)) if match else None

                if light_intensity is not None: # Classify files: zero intensity → dark, non-zero → light
                    if light_intensity == 0:
                        self.dark_files.append(path)
                    elif light_intensity > 0:
                        self.light_files.append(path)
            except Exception as e:
                # If file can’t be read or parsed, skip and log the failure
                print(f"Failed to classify file {path}: {e}")
                continue
