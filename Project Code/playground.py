import pandas as pd
import re

df = pd.read_csv(r'C:\Users\Yoked\Desktop\Thesis Work\Data\8-Pixel Device\back- & forward\devices\sample 1[1]\sample 1[1]_0_1_Perform parallel JV.csv', encoding='latin1', sep=';')

# Extract sample area from the '#Sample area:' metadata line
sample_area_text = df[df.iloc[:, 0].str.contains(r'#Sample area:', na=False)].iloc[0, 0]
match = re.search(r'#Sample area:\t([\d\.]+)', sample_area_text)
sample_area = float(match.group(1)) if match else None

print(sample_area)