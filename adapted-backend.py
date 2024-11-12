import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ETree
import control
import scipy as sc
import streamlit.components.v1 as components




uploaded_file_list = st.file_uploader("Choose your XML file(s)", type="xml",accept_multiple_files=True)
