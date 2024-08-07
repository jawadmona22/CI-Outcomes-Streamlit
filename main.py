import numpy as np
import pandas as pd
from scipy import stats
import scipy
import xml.etree.ElementTree as ETree
from lxml import etree
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# display(HTML("<style>.container { width:100% !important; }</style>"))

xml_file = 'CLIN235 - 1ICE22.xml'

#Load and Parse the Input
Tree = ETree.parse(xml_file)
root = Tree.getroot()

for child in root:
    print(child.tag,child.attrib)

#### SHEET 1 - ECOCHG Summary
A = []  # Assign empty list to use later
for elem in root.iter('Measurements'):
    for subelem in elem:
        B = {}
        for i in list(subelem):
            B.update({i.tag: i.text})
            A.append(B)

ECochG_Series = pd.DataFrame(A)  # Creating a dataframe..(it is a 2 dimensional data structure)

# consists of 3 components ... data, rows, and columns
ECochG_Series.drop_duplicates(keep='first', inplace=True)  # Delete if any Duplicates
ECochG_Series.reset_index(drop=True, inplace=True)  # Reset index after delting Duplicates
ECochG_Series.index = ECochG_Series.index + 1

print(f'Columns present in ECochG_Series Dataframe: {ECochG_Series.columns}')

#################################### PULLING OUT TRACINGS  ####################################

A = []  # Assign empty list to use later
AA = []
for elem in root.iter('Traces'):
    for subelem in elem:
        B = {}
        C = {}
        for i in list(subelem):
            B.update({i.tag: i.text})
            A.append(B)
        C.update(subelem.attrib)
        AA.append(C)

df = pd.DataFrame(A)
df.drop_duplicates(keep='first', inplace=True)  # Delete if any Duplicates
df.reset_index(drop=True, inplace=True)  # Reset index after delting Duplicates

df1 = pd.DataFrame(AA)
df2 = df.join(df1)
df2 = df2.loc[df2["PlotType"] == 'TIME']
df2 = df2.drop('PlotType', axis=1)

# Reindex Dataframe
df2 = df2.reindex(columns=['TraceType', 'X', 'Y'])
### Parsing out attributes from FILE
condensation = df2[df2['TraceType'] == 'CONDENSATION'].reset_index()
rarefaction = df2[df2['TraceType'] == 'RAREFACTION'].reset_index()
Sum = df2[df2['TraceType'] == 'SUM'].reset_index()
difference = df2[df2['TraceType'] == 'DIFFERENCE'].reset_index()


########## PULL CONDENSATION, RAREFACTION, SUM, DIFFERENCE   ##################
###################### PULLING CONDENSATION ######################
condensation1 = condensation['Y'].str.split(" ", expand=True)
condensationY = condensation1.T.unstack()

condensation2 = condensation['X'].str.split(" ", expand=True)
condensationX = condensation2.T.unstack()

condensationX = pd.DataFrame(condensationX)
condensationY = pd.DataFrame(condensationY)

condensation = pd.merge(condensationX, condensationY, left_index=True, right_index=True)
condensation = condensation.rename(columns={"0_x":"Time(us)", "0_y":"Sample(uV)"})
condensation = condensation.reset_index()
condensation = condensation.drop(columns=['level_1'])
condensation = condensation.rename(columns={"level_0":"Measurement Number"})
condensation['Measurement Number']= condensation['Measurement Number'] + 1

###################### PUlLING RAREFACTION  ######################
rarefaction1 = rarefaction['Y'].str.split(" ", expand=True)
rarefactionY = rarefaction1.T.unstack()

rarefaction2 = rarefaction['X'].str.split(" ", expand=True)
rarefactionX = rarefaction2.T.unstack()

rarefactionX = pd.DataFrame(rarefactionX)
rarefactionY = pd.DataFrame(rarefactionY)

rarefaction = pd.merge(rarefactionX, rarefactionY, left_index=True, right_index=True)
rarefaction = rarefaction.rename(columns={"0_x":"Time(us)", "0_y":"Sample(uV)"})
rarefaction = rarefaction.reset_index()
rarefaction = rarefaction.drop(columns=['level_1'])
rarefaction = rarefaction.rename(columns={"level_0":"Measurement Number"})
rarefaction['Measurement Number']= rarefaction['Measurement Number'] + 1

################### PULLING SUM & DIFFERENCE ########################
Sum1 = Sum['Y'].str.split(" ", expand=True)
SumY = Sum1.T.unstack()

Sum2 = Sum['X'].str.split(" ", expand=True)
SumX = Sum2.T.unstack()

SumX = pd.DataFrame(SumX)
SumY = pd.DataFrame(SumY)

Sum = pd.merge(SumX, SumY, left_index=True, right_index=True)
Sum = Sum.rename(columns={"0_x":"Time(us)", "0_y":"Sample(uV)"})
Sum = Sum.reset_index()
Sum = Sum.drop(columns=['level_1'])
Sum = Sum.rename(columns={"level_0":"Measurement Number"})
Sum['Measurement Number']= Sum['Measurement Number'] + 1

difference1 = difference['Y'].str.split(" ", expand=True)
differenceY = difference1.T.unstack()

difference2 = difference['X'].str.split(" ", expand=True)
differenceX = difference2.T.unstack()

differenceX = pd.DataFrame(differenceX)
differenceY = pd.DataFrame(differenceY)
difference = pd.merge(differenceX, differenceY, left_index=True, right_index=True)
difference = difference.rename(columns={"0_x":"Time(us)", "0_y":"Sample(uV)"})
difference = difference.reset_index()
difference = difference.drop(columns=['level_1'])
difference = difference.rename(columns={"level_0":"Measurement Number"})
difference['Measurement Number']= difference['Measurement Number'] + 1


## OUTPUT TO EXCEL
with pd.ExcelWriter("ECOCHG.xlsx") as writer:
    ECochG_Series.to_excel(writer, sheet_name="ECOGHG SERIES", index=True, index_label='Measurement Number')
    difference.to_excel(writer, sheet_name="DIFFERENCE", index=False)
    Sum.to_excel(writer, sheet_name="SUM", index=False)
    condensation.to_excel(writer, sheet_name="CONDENSATION", index=False)
    rarefaction.to_excel(writer, sheet_name="RAREFACTION", index=False)


from numpy import nan
# work with difference function
# Set Parameter (1=250Hz, 2=500Hz, 3=1000Hz, 4=2000Hz)
x_p=2
df = difference.loc[difference['Measurement Number'] == x_p]
df = df.astype("float")
df = df.dropna()

import matplotlib.pyplot as plt
# %matplotlib inline
time = df['Time(us)']*100
voltage = df['Sample(uV)']
plt.title(f"R-C Difference Plotted for Measurement Number {x_p}")
plt.xlabel('Time (us)')
plt.ylabel("Sample (uV)")
plt.plot(time,voltage)
plt.show()


print(f'Voltage: {voltage}')
print(f'time:{time}')

df = pd.DataFrame(voltage)

# Save the DataFrame to an Excel file
filename = 'voltage.xlsx'

df.to_excel(filename, index=False)


import numpy as np
import matplotlib.pyplot as plt

differencefft250 = voltage

# Sampling frequency
Fs250 = 20900  # Hz

# Signal length
L250 = len(differencefft250)

# Zero-padding length
NFFT250 = 2**(int(np.ceil(np.log2(L250))) + 2)

# Compute the FFT and normalize
Y250 = np.fft.fft(differencefft250, NFFT250) / L250

# Frequency vector
f250 = Fs250 / 2 * np.linspace(0, 1, NFFT250 // 2 + 1)


# Convert array to DataFrame
amplitude =  2 * np.abs(Y250[:NFFT250 // 2 + 1])
df = pd.DataFrame(amplitude)

# Save the DataFrame to an Excel file
filename = 'amplitude_500.xlsx'

df.to_excel(filename, index=False)

# Plot the amplitude spectrum
plt.plot(f250, amplitude, linewidth=1.0)
plt.xlim([0, 9000])
plt.title('Fast Fourier Transformation 500 Hz of Difference')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (microvolts)')

# Show the plot
plt.show()
