import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xml.etree.ElementTree as ETree
import scipy as sc
import re

# ── global matplotlib style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "Arial",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
})

BLUE   = "#378ADD"
RED    = "#E24B4A"
GREEN  = "#1D9E75"
AMBER  = "#BA7517"
GRAY   = "#aaaaaa"

PLOT_COLORS = [BLUE, "#D4537E", GREEN, AMBER]  # multi-freq markers

# ── shared CSS ────────────────────────────────────────────────────────────────
SHARED_CSS = """
<style>
  .metric-card {
    background: #f5f5f3;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    min-width: 110px;
  }
  .metric-label {
    font-size: 11px;
    color: #888;
    margin-bottom: 2px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .metric-value {
    font-size: 20px;
    font-weight: 500;
    color: #1a1a1a;
    font-family: Arial;
  }
  .section-label {
    font-size: 11px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 0.25rem;
    margin-top: 0.5rem;
  }
  .plot-note {
    font-size: 10px;
    color: #aaa;
    margin-top: 2px;
    font-family: Arial;
  }
  .streamlit-expanderContent { padding-top: 0.5rem !important; }
  div[data-testid="stHorizontalBlock"] .stButton button {
    border-radius: 20px;
    font-size: 12px;
    padding: 2px 12px;
    height: auto;
  }
</style>
"""

# ── helper: metric card row ───────────────────────────────────────────────────
def metric_row(cards: list[dict]) -> str:
    """cards = [{"label": str, "value": str, "color": optional hex}]"""
    inner = ""
    for c in cards:
        color_style = f'style="color:{c["color"]};"' if c.get("color") else ""
        inner += f"""
          <div class="metric-card">
            <div class="metric-label">{c['label']}</div>
            <div class="metric-value" {color_style}>{c['value']}</div>
          </div>"""
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:0.75rem 0 1rem;">{inner}</div>'


def section_label(text: str):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def plot_note(text: str):
    st.markdown(f'<div class="plot-note">{text}</div>', unsafe_allow_html=True)


# ── shared analysis helpers ───────────────────────────────────────────────────
def natural_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def snr_color(snr_db):
    if snr_db > 6:   return GREEN
    if snr_db > 0:   return AMBER
    return RED


def compute_fft(voltage, fs):
    L = len(voltage)
    NFFT = 2 ** (int(np.ceil(np.log2(L))) + 3)
    Y = np.fft.fft(voltage - np.mean(voltage), NFFT) / L
    freq_array = fs / 2 * np.linspace(0, 1, NFFT // 2 + 1)
    amplitude  = 2 * np.abs(Y[: NFFT // 2 + 1])
    return freq_array, amplitude


def peak_and_snr(amplitude, freq_array, stim_freq):
    idx = int(np.argmin(np.abs(freq_array - stim_freq)))
    peak_amp = amplitude[idx]
    lo = amplitude[max(0, idx - 20) : max(0, idx - 5)]
    hi = amplitude[idx + 5 : idx + 20]
    noise_bins = np.concatenate([lo, hi])
    noise_floor = np.mean(noise_bins) if len(noise_bins) else 1e-9
    snr_db = 20 * np.log10(peak_amp / noise_floor) if noise_floor > 0 else 0.0
    return peak_amp, snr_db, idx


def find_nearest_idx(array, value):
    return int(np.argmin(np.abs(np.asarray(array) - value)))


def noise_floor_calculation(first_harmonic_freq, freq_array, amplitude_array):
    bin_width = 62
    num_bins  = 3
    start_freq  = (9 * bin_width) + first_harmonic_freq
    noise_values = []
    for i in range(num_bins):
        b0 = start_freq + i * bin_width
        b1 = b0 + bin_width
        i0 = find_nearest_idx(freq_array, b0)
        i1 = find_nearest_idx(freq_array, b1)
        noise_values.extend(amplitude_array[i0 : i1 + 1])
    noise_values = np.array(noise_values)
    return np.mean(noise_values) + 3 * np.std(noise_values)


def sum_harmonics_by_peak(fft_data, freq_vector, limit):
    cutoff_index     = np.argmin(np.abs(freq_vector - limit))
    fundamental_index = np.argmax(fft_data[1:]) + 1 + cutoff_index
    if fundamental_index > cutoff_index:
        fundamental_index = cutoff_index
    first_harmonic = np.abs(fft_data[cutoff_index:])
    threshold = noise_floor_calculation(
        first_harmonic_freq=first_harmonic[0],
        freq_array=freq_vector,
        amplitude_array=first_harmonic,
    )
    return fundamental_index, threshold


def highlight_zeros(val):
    return "color: red" if val == 0 else "color: black"


# ── shared plot helpers ───────────────────────────────────────────────────────
def plot_waveform(ax, time_x, voltage, xlabel="Time (ms)"):
    ax.plot(time_x, voltage, linewidth=0.8, color=BLUE)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Amplitude (µV)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))


def plot_fft_spectrum(ax, freq_array, amplitude, stim_freq,
                      xlim=8000, marker_label=None, extra_markers=None,
                      show_noise_floor=True):
    """
    extra_markers: list of (freq_hz, amp_value, color, label) for 2nd/3rd harmonics.
    """
    mask = freq_array <= xlim
    fa   = freq_array[mask]
    amp  = amplitude[mask]

    ax.plot(fa, amp, linewidth=0.9, color=BLUE, zorder=2)
    ax.scatter(fa, amp, s=8, color=BLUE, zorder=3, linewidths=0)

    # Primary stimulus peak (red)
    peak_idx = int(np.argmin(np.abs(fa - stim_freq)))
    peak_amp = amp[peak_idx]
    label = marker_label or f"{stim_freq} Hz  {peak_amp:.4f} µV"
    ax.scatter(fa[peak_idx], peak_amp, s=70, color=RED, zorder=5, label=label)

    # Additional harmonic markers
    if extra_markers:
        for (hf, ha, hc, hl) in extra_markers:
            ax.scatter(hf, ha, s=70, color=hc, zorder=5, label=hl)

    # Noise floor
    if show_noise_floor:
        lo = amp[max(0, peak_idx - 20) : max(0, peak_idx - 5)]
        hi = amp[peak_idx + 5 : peak_idx + 20]
        nf = np.mean(np.concatenate([lo, hi])) if len(lo) + len(hi) else 0
        ax.axhline(nf, color=GRAY, linewidth=0.8, linestyle="--",
                   label=f"Noise floor  {nf:.4f} µV")

    ax.set_xlim([0, xlim])
    top = np.max(amp[2:]) if len(amp) > 2 else 1
    ax.set_ylim([0, top * 1.15])
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Amplitude (µV)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, framealpha=0.5)


# ── XML parsing (Cochlear mode) ───────────────────────────────────────────────
def extract_data(df, trace_type):
    df1 = df["Y"].str.split(" ", expand=True)
    dfY = df1.T.unstack()
    df2 = df["X"].str.split(" ", expand=True)
    dfX = df2.T.unstack()
    result_df = pd.merge(pd.DataFrame(dfX), pd.DataFrame(dfY),
                         left_index=True, right_index=True)
    result_df = result_df.rename(columns={"0_x": "Time(us)", "0_y": "Sample(uV)"})
    result_df = result_df.reset_index().drop(columns=["level_1"])
    result_df = result_df.rename(columns={"level_0": "Measurement Number"})
    result_df["Measurement Number"] += 1
    return result_df


def parse_xml(uploaded_file):
    Tree = ETree.parse(uploaded_file)
    root = Tree.getroot()

    A = []
    for elem in root.iter("Measurements"):
        for subelem in elem:
            B = {}
            for i in list(subelem):
                B[i.tag] = i.text
            A.append(B)

    ECochG_Series = pd.DataFrame(A)
    ECochG_Series.drop_duplicates(keep="last", inplace=True)
    ECochG_Series.reset_index(drop=True, inplace=True)
    ECochG_Series.index += 1
    ECochG_Series["Measurement Number"] = ECochG_Series.index
    current_frequency = int(ECochG_Series["Frequency"][1][1:])

    A, AA = [], []
    for elem in root.iter("Traces"):
        for subelem in elem:
            B, C = {}, {}
            for i in list(subelem):
                B[i.tag] = i.text
                A.append(B)
            C.update(subelem.attrib)
            AA.append(C)

    df  = pd.DataFrame(A).drop_duplicates(keep="first").reset_index(drop=True)
    df2 = df.join(pd.DataFrame(AA))
    df2 = df2.loc[df2["PlotType"] == "TIME"].drop("PlotType", axis=1)
    df2 = df2.reindex(columns=["TraceType", "X", "Y"])

    condensation = df2[df2["TraceType"] == "CONDENSATION"].reset_index()
    rarefaction  = df2[df2["TraceType"] == "RAREFACTION"].reset_index()
    Sum          = df2[df2["TraceType"] == "SUM"].reset_index()
    difference   = df2[df2["TraceType"] == "DIFFERENCE"].reset_index()

    return (extract_data(condensation, "CONDENSATION"),
            extract_data(rarefaction,  "RAREFACTION"),
            extract_data(Sum,          "SUM"),
            extract_data(difference,   "DIFFERENCE"),
            ECochG_Series, current_frequency)


# ── BFTR helpers ──────────────────────────────────────────────────────────────
def extract_max_amplitude(dataframe):
    highest = 0
    recording_electrode = "N/A"
    bftr = ["N/A", 1]
    filtered_df = dataframe[dataframe["Include_In_BFTR"] == True]
    for item in filtered_df["Fundamental Amplitude"]:
        if item > highest:
            highest = item
            recording_electrode = filtered_df.loc[
                filtered_df["Fundamental Amplitude"] == highest, "Recording Electrode"
            ].values
            bftr = filtered_df.loc[
                filtered_df["Fundamental Amplitude"] == highest, "Total Amplitude"
            ].values
    return highest, recording_electrode, bftr


def set_background(color):
    st.markdown(
        f"<style>.stApp {{ background-color: {color}; }}</style>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ECochG Bedside Tool", layout="wide")
st.markdown(SHARED_CSS, unsafe_allow_html=True)
st.title("Electrocochleography Bedside Tool")
plt.close("all")

mode = st.sidebar.radio("Select processing mode:", ["Cochlear", "Advanced Bionics", "IHS"])


# ═══════════════════════════════════════════════════════════════════════════════
# IHS MODE
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "IHS":
    set_background("#F0FAF5")
    st.subheader("Current mode: IHS")

    STIM_FREQS = [250, 500, 750, 1000, 2000, 4000]
    FS = 40000

    uploaded_file_list = st.file_uploader(
        "Choose your text file(s)", type="txt", accept_multiple_files=True
    )

    if uploaded_file_list:
        uploaded_file_list = sorted(uploaded_file_list, key=lambda f: natural_key(f.name))
        st.subheader("Individual File Analysis")

        for uploaded_file in uploaded_file_list:
            with st.expander(f"📄 {uploaded_file.name}", expanded=True):

                df = pd.read_csv(uploaded_file, sep=",", skiprows=24)
                n_samples = len(df["Data Pnt:"])

                time_window = st.slider(
                    "Time window (µs)",
                    min_value=0, max_value=n_samples,
                    value=(int(n_samples * 0.14), int(n_samples * 0.86)),
                    step=50, key="slider_" + uploaded_file.name,
                )
                start_us, end_us = time_window
                start_idx = max(0, min(int(start_us * FS / 1_000_000), n_samples - 1))
                end_idx   = max(start_idx + 1, min(int(end_us * FS / 1_000_000), n_samples))

                voltage_full = df["Average(uV):"].values
                voltage = voltage_full[start_idx:end_idx]
                time_ms = np.linspace(start_idx / FS * 1000, end_idx / FS * 1000, len(voltage))

                # Stimulus selector
                section_label("Stimulus frequency")
                sel_cols = st.columns(len(STIM_FREQS))
                sel_key  = f"stim_ihs_{uploaded_file.name}"
                if sel_key not in st.session_state:
                    st.session_state[sel_key] = 250
                for col, freq in zip(sel_cols, STIM_FREQS):
                    if col.button(f"{freq} Hz", key=f"ihs_btn_{uploaded_file.name}_{freq}",
                                  type="primary" if st.session_state[sel_key] == freq else "secondary"):
                        st.session_state[sel_key] = freq
                stim_freq = st.session_state[sel_key]

                freq_array, amplitude = compute_fft(voltage, fs=FS)
                peak_amp, snr_db, _  = peak_and_snr(amplitude, freq_array, stim_freq)

                st.markdown(metric_row([
                    {"label": "Stim. freq.",      "value": f"{stim_freq} Hz"},
                    {"label": "CM amplitude",     "value": f"{peak_amp:.4f} µV"},
          
                ]), unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    section_label("Cochlear microphonic")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    plot_waveform(ax, time_ms, voltage)
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                with col2:
                    section_label("FFT amplitude spectrum")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    plot_fft_spectrum(ax, freq_array, amplitude, stim_freq)
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                plot_note("fs = 40 kHz · DC removed · 0–8 kHz shown · red marker = stimulus frequency bin")


# ═══════════════════════════════════════════════════════════════════════════════
# COCHLEAR MODE
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "Cochlear":
    set_background("#ffffed")
    st.subheader("Current mode: Cochlear")

    FS_COCHLEAR = 20900

    uploaded_file_list = st.file_uploader(
        "Choose your XML file(s)", type="xml", accept_multiple_files=True
    )

    if uploaded_file_list:
        uploaded_file_list = sorted(uploaded_file_list, key=lambda f: natural_key(f.name))
        st.subheader("Individual File Analysis")
        harmonics_df_dict = {}

        for uploaded_file in uploaded_file_list:
            with st.expander(f"📄 {uploaded_file.name}", expanded=True):

                (condensation_data, rarefaction_data,
                 sum_data, difference_data,
                 ECochG_Series, current_frequency) = parse_xml(uploaded_file)

                unique_meas = difference_data["Measurement Number"].unique()
                harmonic_data = []
                seen_02 = False

                time_window = st.slider(
                    "Time window (µs)",
                    min_value=0, max_value=45000,
                    value=(1992, 16789), step=50,
                    key="slider_" + uploaded_file.name,
                )
                start_time, end_time = time_window

                for x_p in unique_meas:
                    df = (difference_data
                          .loc[difference_data["Measurement Number"] == x_p]
                          .astype("float").dropna())
                    df = df[(df["Time(us)"] >= start_time) & (df["Time(us)"] <= end_time)]

                    sf = (sum_data
                          .loc[sum_data["Measurement Number"] == x_p]
                          .astype("float").dropna())
                    sf = sf[(sf["Time(us)"] > start_time) & (sf["Time(us)"] <= end_time)]

                    subset = ECochG_Series.loc[ECochG_Series["Measurement Number"] == x_p]
                    if subset.empty:
                        continue

                    recording_electrode = ECochG_Series.loc[
                        ECochG_Series["Measurement Number"] == x_p,
                        "RecordingActiveElectrode"
                    ].values[0]
                    include = not seen_02
                    if recording_electrode == "ICE02":
                        seen_02 = True

                    d_time    = df["Time(us)"]
                    d_voltage = np.asarray(df["Sample(uV)"], dtype=np.float64)
                    s_time    = sf["Time(us)"]
                    s_voltage = np.asarray(sf["Sample(uV)"], dtype=np.float64)

                    # FFT
                    freq_array, amplitude   = compute_fft(d_voltage, fs=FS_COCHLEAR)
                    s_freq_array, s_amplitude = compute_fft(s_voltage, fs=FS_COCHLEAR)

                    fundamental_index, threshold = sum_harmonics_by_peak(
                        amplitude, freq_array, current_frequency
                    )
                    fundamental_freq = freq_array[fundamental_index]
                    fundamental_amp  = amplitude[fundamental_index] if amplitude[fundamental_index] > threshold else 0

                    second_harmonic_index = 2 * fundamental_index
                    if second_harmonic_index < len(s_freq_array):
                        second_harmonic_freq = s_freq_array[second_harmonic_index]
                        second_harmonic_amp  = s_amplitude[second_harmonic_index] if s_amplitude[second_harmonic_index] > threshold else 0
                    else:
                        second_harmonic_freq = None
                        second_harmonic_amp  = 0

                    third_harmonic_index = 3 * fundamental_index
                    if third_harmonic_index < len(freq_array):
                        third_harmonic_freq = freq_array[third_harmonic_index]
                        third_harmonic_amp  = amplitude[third_harmonic_index] if amplitude[third_harmonic_index] > threshold else 0
                    else:
                        third_harmonic_freq = None
                        third_harmonic_amp  = 0

                    total_amp = fundamental_amp + second_harmonic_amp + third_harmonic_amp
                    peak_amp_diff, snr_db_diff, _ = peak_and_snr(amplitude, freq_array, current_frequency)

                    # Metric cards
                    st.markdown(metric_row([
                        {"label": "Electrode",       "value": recording_electrode},
                        {"label": "F0 amplitude",    "value": f"{fundamental_amp:.4f} µV"},
                        {"label": "SNR (diff.)",     "value": f"{'+'if snr_db_diff>=0 else ''}{snr_db_diff:.1f} dB",
                                                      "color": snr_color(snr_db_diff)},
                        {"label": "Total amplitude", "value": f"{total_amp:.4f} µV"},
                    ]), unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        section_label(f"R–C difference · electrode {recording_electrode}")
                        fig, ax = plt.subplots(figsize=(6, 3))
                        plot_waveform(ax, d_time / 1000, d_voltage)
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                        section_label(f"R–C sum · electrode {recording_electrode}")
                        fig, ax = plt.subplots(figsize=(6, 3))
                        plot_waveform(ax, s_time / 1000, s_voltage)
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                    with col2:
                        section_label(f"Difference FFT · electrode {recording_electrode}")
                        extra = []
                        if third_harmonic_freq:
                            extra.append((
                                third_harmonic_freq,
                                amplitude[third_harmonic_index],
                                "#534AB7",
                                f"3rd harmonic: {int(third_harmonic_freq)} Hz",
                            ))
                        fig, ax = plt.subplots(figsize=(6, 3))
                        plot_fft_spectrum(
                            ax, freq_array, amplitude, current_frequency,
                            xlim=9000,
                            marker_label=f"F0: {int(fundamental_freq)} Hz  {fundamental_amp:.4f} µV",
                            extra_markers=extra,
                        )
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                        section_label(f"Sum FFT · electrode {recording_electrode}")
                        extra_sum = []
                        if second_harmonic_freq:
                            extra_sum.append((
                                second_harmonic_freq,
                                s_amplitude[second_harmonic_index],
                                GREEN,
                                f"2nd harmonic: {int(second_harmonic_freq)} Hz",
                            ))
                        fig, ax = plt.subplots(figsize=(6, 3))
                        plot_fft_spectrum(
                            ax, s_freq_array, s_amplitude, current_frequency,
                            xlim=9000,
                            marker_label=f"F0: {int(fundamental_freq)} Hz",
                            extra_markers=extra_sum,
                        )
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                    plot_note(
                        f"fs = {FS_COCHLEAR} Hz · DC removed · 0–9 kHz shown · "
                        "red = F0 · purple = F2 · green = F1"
                    )
                    st.divider()

                    harmonic_data.append({
                        "Measurement Number":        x_p,
                        "Recording Electrode":       recording_electrode,
                        "Fundamental Frequency (Hz)": fundamental_freq,
                        "Fundamental Amplitude":      fundamental_amp,
                        "Second Harmonic Freq (Hz)":  second_harmonic_freq,
                        "Second Harmonic Amplitude":  second_harmonic_amp,
                        "Third Harmonic Freq (Hz)":   third_harmonic_freq,
                        "Third Harmonic Amplitude":   third_harmonic_amp,
                        "Total Amplitude":            total_amp,
                        "Include_In_BFTR":            include,
                    })

                harmonic_df = pd.DataFrame(harmonic_data)
                harmonics_df_dict[uploaded_file.name] = harmonic_df

                section_label("Harmonic data")
                st.caption("Red 0s indicate values below the noise floor threshold")
                st.dataframe(harmonic_df.style.applymap(highlight_zeros))

                # Electrode amplitude plot
                section_label("F0 amplitude by electrode")
                def extract_number(name):
                    m = re.search(r"\d+", str(name))
                    return int(m.group()) if m else -1

                df_sorted = harmonic_df.copy()
                df_sorted["_n"] = df_sorted["Recording Electrode"].apply(extract_number)
                df_sorted = df_sorted.sort_values("_n", ascending=False)

                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(df_sorted["Recording Electrode"], df_sorted["Fundamental Amplitude"],
                        "-o", color=BLUE, markersize=6, linewidth=1.2)
                ax.set_title(
                    f"F0 amplitude vs electrode — {int(harmonic_df['Fundamental Frequency (Hz)'].iloc[0])} Hz",
                    fontsize=11
                )
                ax.set_xlabel("Recording electrode", fontsize=10)
                ax.set_ylabel("Amplitude (µV)", fontsize=10)
                ax.tick_params(axis="x", rotation=45, labelsize=9)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            plt.close("all")

        # BFTR summary
        st.subheader("Best Frequency Total Response")
        max_amplitude_dict = {k: extract_max_amplitude(v) for k, v in harmonics_df_dict.items()}
        bftr_rows = []
        for filename, values in max_amplitude_dict.items():
            bftr_rows.append({
                "Filename": filename,
                "BFTR (F0+F1+F3)": values[2][0],
                "Electrode": values[1],
            })
        st.table(pd.DataFrame(bftr_rows))

        # Cross-file electrode plot
        fig, axes = plt.subplots(len(uploaded_file_list), 1,
                                  figsize=(8, 4 * len(uploaded_file_list)))
        axes = np.atleast_1d(axes)

        def extract_number_str(name):
            m = re.search(r"\d+", str(name))
            return int(m.group()) if m else -1

        for i, (key, df_h) in enumerate(harmonics_df_dict.items()):
            df_s = df_h.copy()
            df_s["_n"] = df_s["Recording Electrode"].apply(extract_number_str)
            df_s = df_s.sort_values("_n", ascending=False)
            axes[i].plot(df_s["Recording Electrode"], df_s["Fundamental Amplitude"],
                         "-o", color=BLUE, markersize=6, linewidth=1.2)
            axes[i].set_title(
                f"F0 amplitude vs electrode — {int(df_s['Fundamental Frequency (Hz)'].iloc[0])} Hz",
                fontsize=11
            )
            axes[i].set_xlabel("Recording electrode", fontsize=10)
            axes[i].set_ylabel("Amplitude (µV)", fontsize=10)
            axes[i].tick_params(axis="x", rotation=45, labelsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED BIONICS MODE
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "Advanced Bionics":
    set_background("#f0f8ff")
    st.subheader("Current mode: AB")

    FS_AB = 9280.30303
    ELECTRODES = list(range(1, 17))

    uploaded_file_list = st.file_uploader(
        "Choose your Excel file(s)", type="xlsx", accept_multiple_files=True
    )
    is_multiple_frequencies = st.checkbox("My file(s) contain multiple frequencies", value=True)
    if not is_multiple_frequencies:
        st.warning("Ensure your file(s) contain only one frequency, or check the box above.")

    if uploaded_file_list:
        uploaded_file_list = sorted(uploaded_file_list, key=lambda f: natural_key(f.name))
        st.subheader("Individual File Analysis")
        harmonics_df_dict = {}

        for uploaded_file in uploaded_file_list:
            with st.expander(f"📄 {uploaded_file.name}", expanded=True):

                time_window = st.slider(
                    "Time window (µs)",
                    min_value=0, max_value=1_000_000,
                    value=(281_050, 718_100), step=50,
                    key="slider_" + uploaded_file.name,
                )
                start_time, end_time = time_window

                metadata = pd.read_excel(uploaded_file, nrows=31)
                freqs_present = (metadata[metadata["Settings"] == "RequestedFrequencies"]
                                 .drop("Settings", axis=1).values.flatten()[1:5])
                freqHz = metadata["Unnamed: 2"][10]

                df_raw = pd.read_excel(uploaded_file, skiprows=35)

                # ── single-frequency branch ───────────────────────────────────
                if not is_multiple_frequencies:
                    CM_data = df_raw[df_raw["Type"] == "CM"].copy()
                    CM_data["Electrode Number"] = CM_data["Unnamed: 9"]
                    harmonic_data = []

                    for electrode in ELECTRODES:
                        sub_df  = CM_data[CM_data["Electrode Number"] == electrode].iloc[:, 53:-2]
                        voltage = sub_df.to_numpy().ravel()

                        spm = len(voltage) / 1000
                        mps = 1000 / len(voltage)
                        v_start = int((start_time / 1000) * spm)
                        v_end   = int((end_time   / 1000) * spm)
                        voltage = voltage[v_start:v_end]
                        time_x  = np.linspace(v_start * mps, v_end * mps, len(voltage))

                        freq_array, amplitude = compute_fft(voltage, fs=FS_AB)
                        fundamental_index, threshold = sum_harmonics_by_peak(amplitude, freq_array, freqHz)
                        max_amp      = np.max(amplitude[max(0, fundamental_index-15):fundamental_index+15])
                        fundamental_amp = amplitude[fundamental_index] if amplitude[fundamental_index] > threshold else 0
                        _, snr_db, _ = peak_and_snr(amplitude, freq_array, freqHz)

                        st.markdown(metric_row([
                            {"label": "Electrode",    "value": str(electrode)},
                            {"label": "F0 amplitude", "value": f"{max_amp:.4f} µV"},
                            {"label": "SNR",          "value": f"{'+'if snr_db>=0 else ''}{snr_db:.1f} dB",
                                                       "color": snr_color(snr_db)},
                        ]), unsafe_allow_html=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            section_label(f"CM · electrode {electrode}")
                            fig, ax = plt.subplots(figsize=(6, 3))
                            plot_waveform(ax, time_x, voltage)
                            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                        with col2:
                            section_label(f"FFT · electrode {electrode}")
                            fig, ax = plt.subplots(figsize=(6, 3))
                            plot_fft_spectrum(ax, freq_array, amplitude, freqHz,
                                             xlim=4500,
                                             marker_label=f"F0: {int(freqHz)} Hz  {max_amp:.4f} µV")
                            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                        plot_note(f"fs = {FS_AB:.0f} Hz · DC removed · 0–4500 Hz shown")
                        st.divider()

                        # third harmonic
                        third_harmonic_index = 3 * fundamental_index
                        if third_harmonic_index < len(freq_array):
                            third_harmonic_freq = freq_array[third_harmonic_index]
                            third_harmonic_amp  = amplitude[third_harmonic_index] if amplitude[third_harmonic_index] > threshold else 0
                        else:
                            third_harmonic_freq = None
                            third_harmonic_amp  = 0

                        harmonic_data.append({
                            "Recording Electrode":        electrode,
                            "Fundamental Frequency (Hz)": freqHz,
                            "Fundamental Amplitude":      max_amp,
                            "Second Harmonic Amplitude":  0,
                            "Third Harmonic Freq (Hz)":   third_harmonic_freq,
                            "Third Harmonic Amplitude":   third_harmonic_amp,
                            "Total Amplitude":            fundamental_amp + third_harmonic_amp,
                            "Include_In_BFTR":            True,
                        })

                    harmonic_df = pd.DataFrame(harmonic_data)
                    harmonics_df_dict[uploaded_file.name] = harmonic_df

                    section_label("Harmonic data")
                    st.caption("Red 0s indicate values below the noise floor threshold")
                    st.dataframe(harmonic_df.style.applymap(highlight_zeros))

                    section_label("F0 amplitude by electrode")
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(harmonic_df["Recording Electrode"], harmonic_df["Fundamental Amplitude"],
                            "-o", color=BLUE, markersize=6, linewidth=1.2)
                    ax.set_title(f"F0 amplitude vs electrode — {int(freqHz)} Hz", fontsize=11)
                    ax.set_xlabel("Recording electrode", fontsize=10)
                    ax.set_ylabel("Amplitude (µV)", fontsize=10)
                    ax.tick_params(labelsize=9)
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                # ── multi-frequency branch ────────────────────────────────────
                else:
                    CM_data = df_raw[df_raw["Type"] == "CM"].copy()
                    CM_data["Electrode Number"] = CM_data["Unnamed: 15"]
                    multi_harmonic_data = []

                    for electrode in ELECTRODES:
                        sub_df  = CM_data[CM_data["Electrode Number"] == electrode].iloc[:, 84:]
                        voltage = sub_df.to_numpy().ravel()

                        spm = len(voltage) / 1000
                        mps = 1000 / len(voltage)
                        v_start = int((start_time / 1000) * spm)
                        v_end   = int((end_time   / 1000) * spm)
                        voltage = voltage[v_start:v_end]
                        time_x  = np.linspace(v_start * mps, v_end * mps, len(voltage))

                        freq_array, amplitude = compute_fft(voltage, fs=FS_AB)

                        fundamental_indices, thresholds, max_amps = [], [], []
                        for freq in freqs_present:
                            fi, thr = sum_harmonics_by_peak(amplitude, freq_array, freq)
                            fundamental_indices.append(fi)
                            thresholds.append(thr)
                            max_amps.append(np.max(amplitude[max(0, fi-30):fi+30]))

                        # Metric cards — one per frequency
                        card_list = [{"label": "Electrode", "value": str(electrode)}]
                        for idx_f, freq in enumerate(freqs_present):
                            _, snr_db_f, _ = peak_and_snr(amplitude, freq_array, freq)
                            card_list.append({
                                "label": f"F0 @ {int(freq)} Hz",
                                "value": f"{max_amps[idx_f]:.3f} µV",
                            })
                        st.markdown(metric_row(card_list), unsafe_allow_html=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            section_label(f"CM · electrode {electrode}")
                            fig, ax = plt.subplots(figsize=(6, 3))
                            plot_waveform(ax, time_x, voltage)
                            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                        with col2:
                            section_label(f"FFT · electrode {electrode}")
                            # Build extra markers for each freq beyond the first
                            markers_list = []
                            for idx_f, freq in enumerate(freqs_present):
                                fi = fundamental_indices[idx_f]
                                fa_val = freq_array[fi] if fi < len(freq_array) else freq
                                markers_list.append((
                                    fa_val,
                                    max_amps[idx_f],
                                    PLOT_COLORS[idx_f % len(PLOT_COLORS)],
                                    f"{int(freq)} Hz: {max_amps[idx_f]:.3f} µV",
                                ))
                            # Use first freq as "primary" marker, rest as extras
                            primary   = markers_list[0]
                            extra_m   = markers_list[1:]

                            fig, ax = plt.subplots(figsize=(6, 3))
                            # Draw full spectrum first, then overlay markers manually
                            mask = freq_array <= 4500
                            ax.plot(freq_array[mask], amplitude[mask], linewidth=0.9, color=BLUE, zorder=2)
                            ax.scatter(freq_array[mask], amplitude[mask], s=8, color=BLUE, zorder=3, linewidths=0)
                            for (hf, ha, hc, hl) in markers_list:
                                ax.scatter(hf, ha, s=70, color=hc, zorder=5, label=hl)
                            lo_idx = max(0, fundamental_indices[0] - 20)
                            hi_idx = fundamental_indices[0] + 20
                            nf = np.mean(amplitude[lo_idx:hi_idx]) if hi_idx < len(amplitude) else 0
                            ax.axhline(nf, color=GRAY, linewidth=0.8, linestyle="--",
                                       label=f"Noise floor {nf:.4f} µV")
                            top = np.max(amplitude[2:mask.sum()]) * 1.15 if mask.sum() > 2 else 1
                            ax.set_xlim([0, 4500]); ax.set_ylim([0, top])
                            ax.set_xlabel("Frequency (Hz)", fontsize=10)
                            ax.set_ylabel("Amplitude (µV)", fontsize=10)
                            ax.tick_params(labelsize=9)
                            ax.legend(fontsize=8, framealpha=0.5)
                            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                        plot_note(f"fs = {FS_AB:.0f} Hz · DC removed · 0–4500 Hz shown")
                        st.divider()

                        row = {"Recording Electrode": electrode}
                        for idx_f, freq in enumerate(freqs_present):
                            row[f"F0 {int(freq)} Hz"] = (
                                max_amps[idx_f] if max_amps[idx_f] > thresholds[idx_f] else 0
                            )
                        multi_harmonic_data.append(row)

                    multi_harmonic_df = pd.DataFrame(multi_harmonic_data)
                    harmonics_df_dict[uploaded_file.name] = multi_harmonic_df

                    section_label("Harmonic data")
                    st.caption("Red 0s indicate values below the noise floor threshold")
                    st.dataframe(multi_harmonic_df.style.applymap(highlight_zeros))

                    for idx_f, freq in enumerate(freqs_present):
                        col_name = f"F0 {int(freq)} Hz"
                        section_label(f"F0 amplitude by electrode — {int(freq)} Hz")
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.plot(multi_harmonic_df["Recording Electrode"],
                                multi_harmonic_df[col_name],
                                "-o", color=PLOT_COLORS[idx_f % len(PLOT_COLORS)],
                                markersize=6, linewidth=1.2)
                        ax.set_title(f"F0 amplitude vs electrode — {int(freq)} Hz", fontsize=11)
                        ax.set_xlabel("Recording electrode", fontsize=10)
                        ax.set_ylabel("Amplitude (µV)", fontsize=10)
                        ax.tick_params(labelsize=9)
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            plt.close("all")