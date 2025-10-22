import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title = "Klasifikasi Lemon",
	page_icon = ":lemon:"
)
model =  joblib.load("model_klasifikasi_lemon.joblib")

st.title(":lemon: KlasifikasiLemon")
st.markdown("Aplikasi machine learning classification untuk memprediksi kualitas lemon")

diameter = st.slider("Diameter", 57.1, 60.2, 48.5)
berat = st.slider("Berat", 105, 118, 80)
tebal_kulit = st.slider("Tebal Kulit", 3.7,3.8,4.6)
kadar_gula = st.slider("Kadar Gula", 8.4, 8.2,7.8)
asal_daerah = st.pills("Asal Daerah", ["California", "Malang", "Medan"], default="California" )
warna = st.pills("Warna", ["Hijau pekat","Kuning kehijauan","Kuning cerah"], default="Hijau pekat")
musim_panen = st.pills("Musim Panen", ["Puncak","Akhir","Awal"], default="Puncak")

if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[diameter,berat,tebal_kulit,kadar_gula,asal_daerah,warna,musim_panen]], columns=["diameter","berat","tebal_kulit","kadar_gula","asal_daerah","warna","musim_panen"])
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat dengan :: oleh **Saskia**")