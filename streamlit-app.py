import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ETree
import control
import scipy as sc
import streamlit.components.v1 as components



def ChangeButtonColour(widget_label, font_color, background_color='transparent'):
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{
                if (elements[i].innerText == '{widget_label}') {{
                    elements[i].style.color ='{font_color}';
                    elements[i].style.background = '{background_color}'
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}", height=0, width=0)


def sum_harmonics_by_peak(fft_data):
    # Take the magnitude of the FFT data (ignore the complex part)
    magnitude = np.abs(fft_data)

    # Find the index of the highest peak (ignoring the DC component at index 0)
    fundamental_index = np.argmax(magnitude[1:]) + 1  # +1 to adjust for skipping index 0

    # First harmonic is at the fundamental frequency index (highest peak)
    first_harmonic = magnitude[fundamental_index]

    # Second harmonic is at 2 times the fundamental frequency index
    second_harmonic_index = 2 * fundamental_index
    second_harmonic = magnitude[second_harmonic_index] if second_harmonic_index < len(magnitude) else 0

    # Third harmonic is at 3 times the fundamental frequency index
    third_harmonic_index = 3 * fundamental_index
    third_harmonic = magnitude[third_harmonic_index] if third_harmonic_index < len(magnitude) else 0

    # Sum of the first, second, and third harmonics
    harmonic_sum = first_harmonic + second_harmonic + third_harmonic

    return harmonic_sum, fundamental_index
    
def extract_data(df, trace_type):
    df1 = df['Y'].str.split(" ", expand=True)
    dfY = df1.T.unstack()
    df2 = df['X'].str.split(" ", expand=True)
    dfX = df2.T.unstack()

    dfX = pd.DataFrame(dfX)
    dfY = pd.DataFrame(dfY)
    result_df = pd.merge(dfX, dfY, left_index=True, right_index=True)
    print(result_df.head())

    result_df = result_df.rename(columns={"0_x": "Time(us)", "0_y": "Sample(uV)"})
    result_df = result_df.reset_index()
    result_df = result_df.drop(columns=['level_1'])
    result_df = result_df.rename(columns={"level_0": "Measurement Number"})
    result_df['Measurement Number'] = result_df['Measurement Number'] + 1

    return result_df
    

def parse_xml(uploaded_file):
        # Parse the XML file
    Tree = ETree.parse(uploaded_file)
    root = Tree.getroot()

    # SHEET 1 - ECOCHG Summary
    A = []  # Assign empty list to use later
    for elem in root.iter('Measurements'):
        for subelem in elem:
            B = {}
            for i in list(subelem):
                B.update({i.tag: i.text})

            # Extract the Recording Active Electrode from the Measurements

            print(f'Contents of B: {B}')
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

    # Extracting data
    condensation_data = extract_data(condensation, 'CONDENSATION')
    rarefaction_data = extract_data(rarefaction, 'RAREFACTION')
    sum_data = extract_data(Sum, 'SUM')
    difference_data = extract_data(difference, 'DIFFERENCE')
    return condensation_data,rarefaction_data,sum_data,difference_data

def extract_max_amplitude(dataframe):
    highest = 0
    for item in dataframe["Total Amplitude"]:
        if item > highest:
            highest = item
            
    return highest


# Streamlit app title
st.title("XML Data Analysis with Streamlit")

# Upload XML file
uploaded_file_list = st.file_uploader("Choose your XML file(s)", type="xml",accept_multiple_files=True)
type = st.selectbox("Analysis Type",["Full Insertion","Round Window"])
mode = type = st.selectbox("Mode",["Annotate","Review"])



if (mode == "Annotate"):
    if len(uploaded_file_list) > 0:
        # Select a measurement number
        st.subheader("Individual File Analysis")
        if (type == "Round Window"):
            condensation_data,rarefaction_data,sum_data,difference_data = parse_xml(uploaded_file_list[0])
            list_xp = [1,2,3,4]
            list_hz = ["200","500","1000","2000"]

            # Create a single figure with 4 subplots
            fig, axs = plt.subplots(2,2, figsize=(10, 20))
            axs = axs.flatten()

            dif_fig, dif_axs = plt.subplots(2,2,figsize=(10,20))
            dif_axs = dif_axs.flatten()
            # fig.tight_layout(pad=4.0)
            harmonic_data = []
            for index, x_p in enumerate(list_xp):
                hz = list_hz[index]
                df = difference_data.loc[difference_data['Measurement Number'] == x_p]
                df = df.astype("float")
                df = df.dropna()

                # Extract time and voltage
                time = df['Time(us)'] * 100
                voltage = df['Sample(uV)']
                # Fast Fourier Transformation
                # st.subheader(f"Fast Fourier Transformation for Measurement {x_p} ({hz} Hz)")
                differencefft250 = voltage

                #Plot the selected measurement (if you need this plot separately, otherwise remove it)
                # .figure(figsize=(10, 6))
                dif_axs[index].set_title(f"R-C Difference \n  {hz} Hz")
                dif_axs[index].set_xlabel('Time (us)')
                dif_axs[index].set_ylabel("Sample (uV)")
                dif_axs[index].plot(time, voltage)

                # Sampling frequency
                Fs250 = 20900  # Hz

                # Signal length
                L250 = len(differencefft250)

                # Zero-padding length
                NFFT250 = 2 ** (int(np.ceil(np.log2(L250))) + 2)

                # Compute the FFT and normalize
                Y250 = np.fft.fft(differencefft250, NFFT250) / L250

                # Frequency vector
                f250 = Fs250 / 2 * np.linspace(0, 1, NFFT250 // 2 + 1)

                # Amplitude
                amplitude = 2 * np.abs(Y250[:NFFT250 // 2 + 1])

                # Find harmonics
                harmonic_sum, fundamental_index = sum_harmonics_by_peak(amplitude)

                # Get the frequencies and amplitudes corresponding to the harmonics
                fundamental_freq = f250[fundamental_index]  # First harmonic frequency
                fundamental_amp = amplitude[fundamental_index]  # Amplitude at the first harmonic

                second_harmonic_freq = f250[2 * fundamental_index] if 2 * fundamental_index < len(f250) else None
                second_harmonic_amp = amplitude[2 * fundamental_index] if 2 * fundamental_index < len(amplitude) else 0

                third_harmonic_freq = f250[3 * fundamental_index] if 3 * fundamental_index < len(f250) else None
                third_harmonic_amp = amplitude[3 * fundamental_index] if 3 * fundamental_index < len(amplitude) else 0

                # Sum of the harmonics
                total_amp = fundamental_amp + second_harmonic_amp + third_harmonic_amp

                # Append the data for this measurement to the list
                harmonic_data.append({
                    'Measurement Number': x_p,
                    'Fundamental Frequency (Hz)': fundamental_freq,
                    'Fundamental Amplitude': fundamental_amp,
                    'Second Harmonic Frequency (Hz)': second_harmonic_freq,
                    'Second Harmonic Amplitude': second_harmonic_amp,
                    'Third Harmonic Frequency (Hz)': third_harmonic_freq,
                    'Third Harmonic Amplitude': third_harmonic_amp,
                    'Total Amplitude': total_amp
                })

                # Plot the FFT data with harmonic lines in the respective subplot
                axs[index].plot(f250, amplitude, linewidth=1.0)
                axs[index].set_xlim([0, 9000])
                axs[index].set_title(f'FFT {hz} Hz of Difference')
                axs[index].set_xlabel('Frequency (Hz)')
                axs[index].set_ylabel('Amplitude (microvolts)')

                axs[index].plot(fundamental_freq, amplitude[fundamental_index], 'r*', markersize=10,
                                label=f'1st Harmonic: {fundamental_freq:.2f} Hz')
                if second_harmonic_freq:
                    axs[index].plot(second_harmonic_freq, amplitude[2 * fundamental_index], 'g*', markersize=10,
                                    label=f'2nd Harmonic: {second_harmonic_freq:.2f} Hz')
                if third_harmonic_freq:
                    axs[index].plot(third_harmonic_freq, amplitude[3 * fundamental_index], 'b*', markersize=10,
                                    label=f'3rd Harmonic: {third_harmonic_freq:.2f} Hz')

                # Show legend for each subplot
                axs[index].legend()

            # After the loop, create the DataFrame from the list of harmonic data

            # Display the DataFrame in Streamlit

            st.pyplot(dif_fig)
            st.subheader("Fourier Transform and Total Response")
            harmonic_df = pd.DataFrame(harmonic_data)
            total_response = sum(harmonic_df["Total Amplitude"])
            db_response = control.mag2db(total_response)
            # st.write(f"Total Response:  \n  {total_response} microvolts   \n   {db_response} dB")
            st.write(harmonic_df)
            st.pyplot(fig)
        else:
            harmonics_df_dict = {}
            for uploaded_file in uploaded_file_list:
                with st.expander(f"File: {uploaded_file.name}"):
                    condensation_data, rarefaction_data, sum_data, difference_data = parse_xml(uploaded_file)
                    list_xp = range(1, 13)  # Measurement numbers from 1 to 12
                    harmonic_data = []

                    # Create a single figure for R-C and FFT graphs
                    for index, x_p in enumerate(list_xp):
                        df = difference_data.loc[difference_data['Measurement Number'] == x_p].astype(
                            "float").dropna()

                        # Extract time and voltage
                        time = df['Time(us)'] * 100
                        voltage = df['Sample(uV)']

                        # Create columns for side-by-side plots
                        col1, col2 = st.columns(2)

                        with col1:
                            # Plot R-C Difference
                            st.subheader(f"R-C Difference Measurement {x_p}")
                            st.line_chart(data=pd.DataFrame({'Time (us)': time, 'Sample (uV)': voltage}).set_index(
                                'Time (us)'))

                        # Fast Fourier Transformation
                        Fs250 = 20900  # Hz
                        L250 = len(voltage)
                        NFFT250 = 2 ** (int(np.ceil(np.log2(L250))) + 2)
                        Y250 = np.fft.fft(voltage, NFFT250) / L250
                        f250 = Fs250 / 2 * np.linspace(0, 1, NFFT250 // 2 + 1)
                        amplitude = 2 * np.abs(Y250[:NFFT250 // 2 + 1])

                        # Find harmonics
                        harmonic_sum, fundamental_index = sum_harmonics_by_peak(amplitude)
                        fundamental_freq = f250[fundamental_index]
                        fundamental_amp = amplitude[fundamental_index]
                        second_harmonic_freq = f250[2 * fundamental_index] if 2 * fundamental_index < len(
                            f250) else None
                        second_harmonic_amp = amplitude[2 * fundamental_index] if 2 * fundamental_index < len(
                            amplitude) else 0
                        third_harmonic_freq = f250[3 * fundamental_index] if 3 * fundamental_index < len(
                            f250) else None
                        third_harmonic_amp = amplitude[3 * fundamental_index] if 3 * fundamental_index < len(
                            amplitude) else 0
                        total_amp = fundamental_amp + second_harmonic_amp + third_harmonic_amp

                        # Append the data for this measurement to the list
                        harmonic_data.append({
                            'Measurement Number': x_p,
                            'Fundamental Frequency (Hz)': fundamental_freq,
                            'Fundamental Amplitude': fundamental_amp,
                            'Second Harmonic Frequency (Hz)': second_harmonic_freq,
                            'Second Harmonic Amplitude': second_harmonic_amp,
                            'Third Harmonic Frequency (Hz)': third_harmonic_freq,
                            'Third Harmonic Amplitude': third_harmonic_amp,
                            'Total Amplitude': total_amp
                        })

                        with col2:
                            # Plot FFT
                            st.subheader(f'FFT Measurement {x_p}')
                            fig_fft, ax_fft = plt.subplots()
                            ax_fft.plot(f250, amplitude, linewidth=1.0)
                            ax_fft.set_xlim([0, 9000])
                            ax_fft.set_title(f'FFT Measurements {x_p}')
                            ax_fft.set_xlabel('Frequency (Hz)')
                            ax_fft.set_ylabel('Amplitude (microvolts)')
                            ax_fft.plot(fundamental_freq, amplitude[fundamental_index], 'r*', markersize=10,
                                        label=f'1st Harmonic: {fundamental_freq:.2f} Hz')
                            if second_harmonic_freq:
                                ax_fft.plot(second_harmonic_freq, amplitude[2 * fundamental_index], 'g*',
                                            markersize=10,
                                            label=f'2nd Harmonic: {second_harmonic_freq:.2f} Hz')
                            if third_harmonic_freq:
                                ax_fft.plot(third_harmonic_freq, amplitude[3 * fundamental_index], 'b*',
                                            markersize=10,
                                            label=f'3rd Harmonic: {third_harmonic_freq:.2f} Hz')
                            ax_fft.legend()
                            st.pyplot(fig_fft)

                    # Create a DataFrame from harmonic data and display it
                    harmonic_df = pd.DataFrame(harmonic_data)
                    st.subheader("Harmonic Data")
                    st.write(harmonic_df)

                    harmonics_df_dict[uploaded_file.name] = harmonic_df

            # Optionally summarize best frequency total response
            st.subheader("Best Frequency Total Response")
            max_amplitude_dict = {key: extract_max_amplitude(value) for key, value in harmonics_df_dict.items()}
            st.write(max_amplitude_dict)





            if st.button('True Response', key='b1'):
                st.write("File recorded!")

            if st.button('Invalid Response', key='b2'):
                st.write("File recorded!")

            ChangeButtonColour('True Response', 'white', 'green') # button txt to find, colour to assign
            ChangeButtonColour('Invalid Response', 'white', 'red') # button txt to find, colour to assign


else:
    st.write("Review mode selected")


