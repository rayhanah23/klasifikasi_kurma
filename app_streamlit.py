
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_icon="🧆",
    page_title="Klasifikasi Kurma"
)

model = joblib.load("model_klasifikasi_kurma.joblib")

st.title("🥧 Klasifikasi Kurma")
st.markdown("Aplikasi machine learning sederhana untuk klasifikasi macam macam kurma")

ukuran_cm = st.slider("Ukuran cm", 0.0, 7.0, 1.70)
berat_g = st.slider("Berat Gram", 0.0, 20.5, 1.70)
kekerasan_N = st.slider("Kekerasan", 0.0, 90.0, 2.00)
kadar_gula_brix = st.slider("Kadar Gula", 0.0, 9.0, 8.0)
kadar_air_pct = st.slider("Kadar Air", 0.0, 15.0, 10.0)

if st.button("Prediksi", type="primary"):
    data = pd.DataFrame(
        [[ukuran_cm, berat_g, kekerasan_N, kadar_gula_brix, kadar_air_pct]],
        columns=[
            "ukuran_cm",
            "berat_g",
            "kekerasan_N",
            "kadar_gula_brix",
            "kadar_air_pct"
        ]
    )

    prediksi = model.predict(data)[0]
    presentase = max(model.predict_proba(data)[0])

    st.success(
        f"Model memprediksi **{prediksi}** "
        f"dengan tingkat keyakinan **{presentase * 100:.2f}%**"
    )
    st.balloons()
    st.snow()

st.divider()
st.caption("Dibuat dengan :coffee: oleh **Rayhanah Salsabila Kamal**")