import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ETree
import scipy

# Streamlit app title
st.title("XML Data Analysis with Streamlit")

# Upload XML file
uploaded_file = st.file_uploader("Choose an XML file", type="xml")

if uploaded_file is not None:
    # Parse the XML file
    Tree = ETree.parse(uploaded_file)
    root = Tree.getroot()

    # Display XML structure
    # st.subheader("XML Structure")
    # xml_structure = ""
    # for child in root:
    #     xml_structure += f"{child.tag} {child.attrib}\n"
    # st.text(xml_structure)

    # SHEET 1 - ECOCHG Summary
    A = []  # Assign empty list to use later
    for elem in root.iter('Measurements'):
        for subelem in elem:
            B = {}
            for i in list(subelem):
                B.update({i.tag: i.text})
            A.append(B)

    ECochG_Series = pd.DataFrame(A)
    ECochG_Series.drop_duplicates(keep='first', inplace=True)
    ECochG_Series.reset_index(drop=True, inplace=True)
    ECochG_Series.index = ECochG_Series.index + 1

    st.subheader("ECochG Series Dataframe")
    st.dataframe(ECochG_Series)

    # PULLING OUT TRACINGS
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
    df.drop_duplicates(keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df1 = pd.DataFrame(AA)
    df2 = df.join(df1)
    df2 = df2.loc[df2["PlotType"] == 'TIME']
    df2 = df2.drop('PlotType', axis=1)

    # Reindex Dataframe
    df2 = df2.reindex(columns=['TraceType', 'X', 'Y'])
    condensation = df2[df2['TraceType'] == 'CONDENSATION'].reset_index()
    rarefaction = df2[df2['TraceType'] == 'RAREFACTION'].reset_index()
    Sum = df2[df2['TraceType'] == 'SUM'].reset_index()
    difference = df2[df2['TraceType'] == 'DIFFERENCE'].reset_index()

    # Extract data function
    def extract_data(df, trace_type):
        df1 = df['Y'].str.split(" ", expand=True)
        dfY = df1.T.unstack()
        df2 = df['X'].str.split(" ", expand=True)
        dfX = df2.T.unstack()

        dfX = pd.DataFrame(dfX)
        dfY = pd.DataFrame(dfY)

        result_df = pd.merge(dfX, dfY, left_index=True, right_index=True)
        result_df = result_df.rename(columns={"0_x": "Time(us)", "0_y": "Sample(uV)"})
        result_df = result_df.reset_index()
        result_df = result_df.drop(columns=['level_1'])
        result_df = result_df.rename(columns={"level_0": "Measurement Number"})
        result_df['Measurement Number'] = result_df['Measurement Number'] + 1

        return result_df

    # Extracting data
    condensation_data = extract_data(condensation, 'CONDENSATION')
    rarefaction_data = extract_data(rarefaction, 'RAREFACTION')
    sum_data = extract_data(Sum, 'SUM')
    difference_data = extract_data(difference, 'DIFFERENCE')

    # Display dataframes
    st.subheader("Condensation Data")
    st.dataframe(condensation_data)

    st.subheader("Rarefaction Data")
    st.dataframe(rarefaction_data)

    st.subheader("Sum Data")
    st.dataframe(sum_data)

    st.subheader("Difference Data")
    st.dataframe(difference_data)

    # Select a measurement number
    st.subheader("Plot R-C Difference")
    # x_p = st.number_input("Select Measurement Number", min_value=1, max_value=len(difference_data), value=1, step=1)
    # Plot the amplitude spectrum
    hz = st.selectbox(
        "Which frequency (in Hz) do you want to analyze?",
        ("200", "500", "1000", "2000"))

    if hz == "200":
        x_p = 1
    if hz == "500":
        x_p = 2
    if hz == "1000":
        x_p = 3
    if hz == "2000":
        x_p = 4

    df = difference_data.loc[difference_data['Measurement Number'] == x_p]
    df = df.astype("float")
    df = df.dropna()

    # Plot the selected measurement
    time = df['Time(us)'] * 100
    voltage = df['Sample(uV)']

    plt.figure(figsize=(10, 6))
    plt.title(f"R-C Difference Plotted for Measurement Number {x_p}")
    plt.xlabel('Time (us)')
    plt.ylabel("Sample (uV)")
    plt.plot(time, voltage)
    st.pyplot(plt)

    # Export Voltage to Excel
    st.subheader("Export Data")
    export_df = pd.DataFrame(voltage)
    export_filename = st.text_input("Export Voltage Data as Excel Filename", value='voltage.xlsx')
    if st.button("Export Voltage Data"):
        export_df.to_excel(export_filename, index=False)
        st.success(f"Data exported successfully to {export_filename}")

    # Fast Fourier Transformation
    st.subheader("Fast Fourier Transformation")
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
    amplitude = 2 * np.abs(Y250[:NFFT250 // 2 + 1])
    amplitude_df = pd.DataFrame(amplitude)

    # Export Amplitude to Excel
    export_amp_filename = st.text_input("Export Amplitude Data as Excel Filename", value='amplitude_500.xlsx')
    if st.button("Export Amplitude Data"):
        amplitude_df.to_excel(export_amp_filename, index=False)
        st.success(f"Data exported successfully to {export_amp_filename}")



    plt.figure(figsize=(10, 6))
    plt.plot(f250, amplitude, linewidth=1.0)
    plt.xlim([0, 9000])
    plt.title(f'Fast Fourier Transformation {hz} Hz of Difference')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (microvolts)')
    st.pyplot(plt)
