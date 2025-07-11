import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ETree
import scipy as sc
import streamlit.components.v1 as components
import re


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx




def noise_floor_calculation(first_harmonic_freq, freq_array, amplitude_array):
    bin_width = 62  # Each bin is 62 Hz wide
    num_bins = 3

    start_freq = (9 * bin_width) + first_harmonic_freq
    noise_values = []

    for i in range(num_bins):
        # Get frequency range for the bin
        bin_start_freq = start_freq + (i * bin_width)
        bin_end_freq = bin_start_freq + bin_width

        # Get index range
        bin_start_idx = find_nearest_idx(freq_array, bin_start_freq)
        bin_end_idx = find_nearest_idx(freq_array, bin_end_freq)

        # Append raw amplitude values in the bin
        noise_values.extend(amplitude_array[bin_start_idx:bin_end_idx + 1])

    # Convert to numpy array in case it isn't
    noise_values = np.array(noise_values)

    # Compute mean and std
    mean_noise = np.mean(noise_values)
    std_noise = np.std(noise_values)

    threshold = mean_noise + 3 * std_noise
    return threshold



def sum_harmonics_by_peak(fft_data,freq_vector, limit):
    # Take the magnitude of the FFT data
    #Find cut-off index of limit (which is the frequency we are testing at, e.g 250 or 500 or 1000)
    #Index isn't exact due to how freq_vector is formed
    cutoff_index = np.argmin(np.abs(freq_vector - limit))
    magnitude = np.abs(fft_data[cutoff_index:])


    # Find the index of the highest peak (ignoring the DC component at index 0)
    fundamental_index = np.argmax(fft_data[1:]) + 1 + cutoff_index # +1 to adjust for skipping index 0, +cutoff index to account for that

    if fundamental_index > cutoff_index: #prevent values above where we are looking
        fundamental_index = cutoff_index

    # First harmonic is at the fundamental frequency index (highest peak)
    first_harmonic = magnitude[fundamental_index]

    #Find the noise floor
    threshold = noise_floor_calculation(first_harmonic_freq=first_harmonic,freq_array=freq_vector,amplitude_array=magnitude)

    return fundamental_index, threshold
    
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
    

def parse_xml(uploaded_file):
    Tree = ETree.parse(uploaded_file)
    root = Tree.getroot()

    A = []  
    for elem in root.iter('Measurements'):
        for subelem in elem:
            B = {}
            for i in list(subelem):
                B.update({i.tag: i.text})
            A.append(B)

    ECochG_Series = pd.DataFrame(A)
    ECochG_Series.drop_duplicates(keep='last', inplace=True)
    ECochG_Series.reset_index(drop=True, inplace=True)
    ECochG_Series.index = ECochG_Series.index + 1
    ECochG_Series['Measurement Number'] = ECochG_Series.index
    current_frequency = int(ECochG_Series['Frequency'][1][1:])
    # Display columns for verification
    # recording_electrode = ECochG_Series.loc[ECochG_Series['RecordingActiveElectrode'] == 'ICE20', 'Measurement Number'].values
    # print(f"Example Recording Electrode: {recording_electrode}")

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
    

    # Extracting data
    condensation_data = extract_data(condensation, 'CONDENSATION')
    rarefaction_data = extract_data(rarefaction, 'RAREFACTION')
    sum_data = extract_data(Sum, 'SUM')
    difference_data = extract_data(difference, 'DIFFERENCE')
 

    return condensation_data,rarefaction_data,sum_data,difference_data,ECochG_Series, current_frequency

def extract_max_amplitude(dataframe):  ##Changed to now extract max fundamental frequency
    highest = 0
    electrode = 0
    filtered_df = dataframe[dataframe["Include_In_BFTR"] == True]

    for item in filtered_df["Fundamental Amplitude"]:

        if item > highest:
            highest = item
            recording_electrode = filtered_df.loc[filtered_df['Fundamental Amplitude'] == highest, 'Recording Electrode'].values
            bftr = filtered_df.loc[filtered_df['Fundamental Amplitude'] == highest, 'Total Amplitude'].values

    return highest, recording_electrode,bftr

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

##########Begin Main Call##############
st.title("Electrocochleography Bedside Tool")
plt.close('all')
# Upload XML file
uploaded_file_list = st.file_uploader("Choose your XML file(s)", type="xml",accept_multiple_files=True)
if len(uploaded_file_list) > 0:
    uploaded_file_list = sorted(uploaded_file_list, key=lambda f: natural_key(f.name))
    # Select a measurement number
    st.subheader("Individual File Analysis")
    harmonics_df_dict = {} #setting up for BFTR calculation
    file_index = 0
    for uploaded_file in uploaded_file_list:
        file_index +=1
        with st.expander(f"File: {uploaded_file.name}"):
            condensation_data, rarefaction_data, sum_data, difference_data,ECochG_Series, current_frequency = parse_xml(uploaded_file)
            # Get unique measurement numbers from 'difference_data'
            unique_measurement_numbers = difference_data['Measurement Number'].unique()
            harmonic_data = []
            seen_02 = False
            #Get indices of time series between 6ms (6000us) and 15ms (15,000 ms)
            time_window = st.slider(
                "Select time window (in microseconds)",
                min_value=0,
                max_value=45000,
                # value=(6000, 15000),
                value=(1992, 16789),
                step=50,
                key=lambda f: natural_key(f.name) + 'f'
            )
            start_time, end_time = time_window


            for index, x_p in enumerate(unique_measurement_numbers):
                df = difference_data.loc[difference_data['Measurement Number'] == x_p].astype(
                    "float").dropna()
                
                df = df[(df["Time(us)"] >= start_time) & (df["Time(us)"] <=end_time)] ##Windowing

                sf = sum_data.loc[sum_data['Measurement Number'] == x_p].astype(
                    "float").dropna()
                
                sf = sf[(sf["Time(us)"] > start_time) & (sf["Time(us)"] <=end_time)]

                
                subset = ECochG_Series.loc[ECochG_Series['Measurement Number'] == x_p]

                if subset.empty:
                    print("The DataFrame is empty.")
                else:
                    recording_electrode = ECochG_Series.loc[ECochG_Series['Measurement Number'] == x_p, 'RecordingActiveElectrode'].values[0]
                    if seen_02:
                        include = False
                    else:
                        include = True

                    if recording_electrode == "ICE02":
                        seen_02 = True


                    ##Difference Data Time Series##

                    # Extract time and voltage
                    d_time = df['Time(us)']
                    
                    d_voltage = df['Sample(uV)']
                    # print("D_Voltage Length", len(d_voltage))
                    # print(d_voltage)
                    d_voltage = np.asarray(d_voltage, dtype=np.float64)

                    # Create columns for side-by-side plots
                    col1, col2 = st.columns(2)

                    with col1:
                        # Plot R-C Difference
                        st.subheader(f"R-C Difference Electrode {recording_electrode}")
                        fig_linechart, ax_linechart = plt.subplots()

                        ax_linechart.plot(d_time/1000, d_voltage, linewidth=1.0)
                        ax_linechart.set_xlabel('Time (ms)')
                        ax_linechart.set_ylabel('Sample (uV)')
                        st.pyplot(fig_linechart)

                
                    
                    ##Sum Data Time Series## 
                    # Extract time and voltage
                    s_time = sf['Time(us)']
                    s_voltage = sf['Sample(uV)']
                    s_voltage = np.asarray(s_voltage, dtype=np.float64)

                    with col1:
                        # Plot R-C Difference
                        st.subheader(f"R-C Sum Electrode {recording_electrode}")
                        fig_linechart, ax_linechart = plt.subplots()

                        ax_linechart.plot(s_time/1000, s_voltage, linewidth=1.0)
                        ax_linechart.set_xlabel('Time (ms)')
                        ax_linechart.set_ylabel('Sample (uV)')
                        st.pyplot(fig_linechart)

                # Difference Fast Fourier Transformation
                Fs250 = 20900  # Hz
                L250 = len(d_voltage) 
    

                NFFT250 = 2 ** (int(np.ceil(np.log2(L250))) + 3)
                # print("NFFT250 ", NFFT250)

                Y250 = np.fft.fft(d_voltage, NFFT250) / L250
                freq_array = Fs250 / 2 * np.linspace(0, 1, NFFT250 // 2 + 1)
                # print(f"Freq Array for Electrode {recording_electrode}", freq_array)
                amplitude = 2 * np.abs(Y250[:NFFT250 // 2 + 1])

                # Sum Fast Fourier Transformation
                SL250 = len(s_voltage)
                SNFFT250 = 2 ** (int(np.ceil(np.log2(SL250))) + 3)
                SY250 = np.fft.fft(s_voltage, NFFT250) / SL250
                s_freq_array = Fs250 / 2 * np.linspace(0, 1, SNFFT250 // 2 + 1)
                s_amplitude = 2 * np.abs(SY250[:SNFFT250 // 2 + 1])


                # Find harmonics after the limit of the file given for DIFFERENCE. 1st and third are from diff, second from SUM.
                fundamental_index, threshold = sum_harmonics_by_peak(amplitude, freq_array,current_frequency)
                # Fundamental frequency and amplitude
                fundamental_freq = freq_array[fundamental_index]
                fundamental_amp = amplitude[fundamental_index] if amplitude[fundamental_index] > threshold else 0

                # Second harmonic frequency and amplitude
                second_harmonic_index = 2 * fundamental_index
                if second_harmonic_index < len(freq_array):
                    second_harmonic_freq = s_freq_array[second_harmonic_index]
                    second_harmonic_amp = s_amplitude[second_harmonic_index] if s_amplitude[second_harmonic_index] > threshold else 0
                else:
                    second_harmonic_freq = None
                    second_harmonic_amp = 0

                # Third harmonic frequency and amplitude
                third_harmonic_index = 3 * fundamental_index
                if third_harmonic_index < len(freq_array):
                    third_harmonic_freq = freq_array[third_harmonic_index]
                    third_harmonic_amp = amplitude[third_harmonic_index] if amplitude[third_harmonic_index] > threshold else 0
                else:
                    third_harmonic_freq = None
                    third_harmonic_amp = 0


                total_amp = fundamental_amp + second_harmonic_amp + third_harmonic_amp

                # Append the data for this measurement to the list
                harmonic_data.append({
                    'Measurement Number': x_p,
                    'Recording Electrode':recording_electrode,
                    'Fundamental Frequency (Hz)': fundamental_freq,
                    'Fundamental Amplitude': fundamental_amp,
                    'Second Harmonic Frequency (Hz)': second_harmonic_freq,
                    'Second Harmonic Amplitude': second_harmonic_amp,
                    'Third Harmonic Frequency (Hz)': third_harmonic_freq,
                    'Third Harmonic Amplitude': third_harmonic_amp,
                    'Total Amplitude': total_amp,
                    'Include_In_BFTR':include
                })

                with col2:
                    # Plot Difference FFT
                    st.subheader(f'FFT Electrode {recording_electrode}')
                    fig_fft, ax_fft = plt.subplots()
                    ax_fft.plot(freq_array, amplitude, linewidth=1.0)
                    ax_fft.set_xlim([0, 9000])
                    ax_fft.set_title(f'FFT Measurement {x_p}')
                    ax_fft.set_xlabel('Frequency (Hz)')
                    ax_fft.set_ylabel('Amplitude (microvolts)')
                    ax_fft.plot(fundamental_freq, amplitude[fundamental_index], 'r*', markersize=10,
                                label=f'1st Harmonic: {int(fundamental_freq)} Hz')

      
                    if third_harmonic_freq:
                        ax_fft.plot(third_harmonic_freq, amplitude[3 * fundamental_index], 'b*',
                                    markersize=10,
                                    label=f'3rd Harmonic: {int(third_harmonic_freq):.2f} Hz')
                    ax_fft.legend()
                    st.pyplot(fig_fft)

                    # Plot Sum FFT
                    st.subheader(f'FFT Electrode {recording_electrode}')
                    fig_fft, ax_fft = plt.subplots()
                    ax_fft.plot(s_freq_array, s_amplitude, linewidth=1.0)
                    ax_fft.set_xlim([0, 9000])
                    ax_fft.set_title(f'FFT Measurement {x_p}')
                    ax_fft.set_xlabel('Frequency (Hz)')
                    ax_fft.set_ylabel('Amplitude (microvolts)')
                    ax_fft.plot(fundamental_freq, s_amplitude[fundamental_index])

                    if second_harmonic_freq:
                        ax_fft.plot(second_harmonic_freq, amplitude[2 * fundamental_index], 'g*',
                                    markersize=10,
                                    label=f'2nd Harmonic: {int(second_harmonic_freq):.2f} Hz')
                  
                    ax_fft.legend()
                    st.pyplot(fig_fft)

            # Create a DataFrame from harmonic data and display it
            harmonic_df = pd.DataFrame(harmonic_data)
            def highlight_zeros(val):
                color = 'red' if val == 0 else 'black'
                return f'color: {color}'

            # Apply the style to all values in the DataFrame
            styled_df = harmonic_df.style.applymap(highlight_zeros)

            # Display with Streamlit
            st.subheader("Harmonic Data")
            st.write("Red 0s indicate where values were below the noise floor threshold")
            st.dataframe(styled_df)

            harmonics_df_dict[uploaded_file.name] = harmonic_df

            fig, ax = plt.subplots()
            ax.plot(harmonic_df["Recording Electrode"][:], harmonic_df["Fundamental Amplitude"][:],'-o')
            ax.set_title(f"Amplitude of F0 vs Electrode for {int(harmonic_df['Fundamental Frequency (Hz)'][0])} Hz")
            ax.set_xlabel("Recording Electrode")
            ax.set_ylabel("Amplitude (uV)")
            ax.set_xticklabels(harmonic_df["Recording Electrode"],rotation=45, ha="right")
            st.pyplot(fig)


    st.subheader("Best Frequency Total Response")
    max_amplitude_dict = {key: extract_max_amplitude(value) for key, value in harmonics_df_dict.items()}
    data = []
    for filename, values in max_amplitude_dict.items():
        response = values[0]  # First value is the response
        electrode = values[1]  # Second value is the electrode
        bftr = values[2][0]
        data.append({"Filename": filename, "BFTR (F0+F1+F3)": bftr, "Electrode": electrode})

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Display the DataFrame in Streamlit
    st.write("Max Amplitude Table")
    st.table(df)

  

    # Create a vertical stack of subplots (one column, multiple rows)
    fig, ax = plt.subplots(len(uploaded_file_list), 1, figsize=(8, 4 * len(uploaded_file_list)))
    ax = np.atleast_1d(ax)  # Ensure ax is always iterable even if only 1 plot


    def extract_number(electrode_name):
        # Extracts the numeric part from something like "ICE22"
        match = re.search(r'\d+', electrode_name)
        return int(match.group()) if match else -1


    for i, (key, df) in enumerate(harmonics_df_dict.items()):
        # Sort by numeric part of electrode name (descending)
        df_sorted = df.copy()
        df_sorted["ElectrodeNumber"] = df_sorted["Recording Electrode"].apply(extract_number)
        df_sorted = df_sorted.sort_values(by="ElectrodeNumber", ascending=False)

        ax[i].plot(df_sorted["Recording Electrode"], df_sorted["Fundamental Amplitude"], '-o')
        ax[i].set_title(f"Amplitude of F0 vs Electrode for {int(df_sorted['Fundamental Frequency (Hz)'].iloc[0])} Hz")
        ax[i].set_xlabel("Recording Electrode")
        ax[i].set_ylabel("Amplitude (uV)")
        ax[i].set_xticks(df_sorted["Recording Electrode"])
        ax[i].set_xticklabels(df_sorted["Recording Electrode"], rotation=45, ha="right")

    plt.tight_layout()
    st.pyplot(fig)


