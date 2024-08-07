import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('model/pipe_object.pkl', 'rb'))
df = pickle.load(open('model/laptop_data.pkl', 'rb'))

st.title('Laptop Price Prediction')

st.subheader("Basic Laptop Info")
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [4, 6, 8, 12, 16, 24, 32, 64], index=2)
weight = st.number_input('Weight of the Laptop(in Kg)', min_value=1.0, max_value=2.8, value=1.8, step=0.1)

st.subheader("Advanced Laptop Info")
touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['Yes', 'No'])
screen_size = st.selectbox('Screen Size (in Inch)', [13.3, 15.6, 17.3])
resolution = st.selectbox('Screen Resolution', ['1920 x 1080', '1366 x 768', '1600 x 900','3840 x 2160', '3200 x 1800', '2880 x 1800','2560 x 1600', '2560 x 1440', '2304 x 1440'])
os = st.selectbox('OS', df['os'].unique())
cpu = st.selectbox('CPU', df['Cpu Brand'].unique())
gpu = st.selectbox('GPU', df['GpuBrand'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 128, 256, 512, 1024])

if st.button('Predict Price'):
    touchscreen = int(touchscreen == 'Yes')
    ips = int(ips == 'Yes')
    X_res, Y_res = map(int, resolution.split(' x ')) 
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, -1)
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    st.title(f"\nPrice: ₹ {round(predicted_price)}")