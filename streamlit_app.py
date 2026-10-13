from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CAD, SUN, LIGHT, BIS, INK = "#ED872D", "#FD5E53", "#FFB766", "#FFE4C4", "#4A2410"
LABELS = {"Age": "Usia", "Study_Hours": "Jam Belajar", "Sleep_Hours": "Jam Tidur",
          "Attendance": "Kehadiran (%)", "Exam_Score": "Nilai Ujian",
          "Monthly_Spending": "Pengeluaran Bulanan", "Screen_Hours": "Jam Layar",
          "Satisfaction": "Kepuasan (1-10)"}
NUM = list(LABELS)

st.set_page_config(page_title="Dashboard Statistik Deskriptif", layout="wide")
st.markdown(f"""<style>
.stApp{{background:linear-gradient(180deg,{BIS},#FFEFD9)}}
h1,h2,h3{{color:{CAD}!important}}
[data-testid=stMetric]{{background:#FFF4E6;border:1px solid #F6CFA0;border-left:6px solid {CAD};
 border-radius:14px;padding:12px 16px}}
[data-testid=stSidebar]{{background:{LIGHT}33}}
.stApp [data-testid=stMarkdownContainer] *,.stApp [data-testid=stWidgetLabel] *,.stApp [data-testid=stCaptionContainer] *,.stApp [data-testid=stMetric] *,.stApp .xtick text,.stApp .ytick text,.stApp .legendtext,.stApp .g-xtitle text,.stApp .g-ytitle text,.stApp .cbaxis text,.stApp .gtitle{{color:{INK}!important;fill:{INK}!important}}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    return pd.read_csv(latihan1.cvs)


df = load()
st.sidebar.header("Filter")
g = st.sidebar.selectbox("Gender", ["Semua", "F", "M"])
a = st.sidebar.selectbox("Usia", ["Semua"] + sorted(df.Age.unique().tolist()))
if g != "Semua":
    df = df[df.Gender == g]
if a != "Semua":
    df = df[df.Age == a]

st.title("Dashboard Statistik Deskriptif - Latihan 1")
if len(df) < 2:
    st.warning("Data terlalu sedikit untuk filter ini.")
    st.stop()

k = st.columns(5)
k[0].metric("Jumlah Mahasiswa", len(df))
k[1].metric("Rata-rata Nilai", f"{df.Exam_Score.mean():.2f}", f"SD {df.Exam_Score.std():.2f}", delta_color="off")
k[2].metric("Rata-rata Kehadiran", f"{df.Attendance.mean():.1f}%")
k[3].metric("Rata-rata Jam Belajar", f"{df.Study_Hours.mean():.2f}")
k[4].metric("Rata-rata Kepuasan", f"{df.Satisfaction.mean():.2f}")

st.subheader("Tabel Statistik Deskriptif")
rows = {}
for c in NUM:
    s = df[c]
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    rows[LABELS[c]] = dict(N=s.count(), Mean=s.mean(), Median=s.median(), Modus=s.mode().iloc[0],
        Std=s.std(), Min=s.min(), Q1=q1, Q3=q3, Max=s.max(), IQR=iqr,
        **{"CV (%)": s.std() / s.mean() * 100, "Skew": s.skew(),
           "Outlier": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())})
st.dataframe(pd.DataFrame(rows).T.round(2), use_container_width=True)

def style(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFF4E6",
                      font_color=INK, margin=dict(t=40, b=10))
    return fig

c1, c2 = st.columns(2)
with c1:
    st.subheader("Distribusi")
    v = st.selectbox("Variabel", NUM, index=NUM.index("Exam_Score"), format_func=LABELS.get)
    st.plotly_chart(style(px.histogram(df, x=v, nbins=8, color_discrete_sequence=[CAD],
                    labels=LABELS)), use_container_width=True)
with c2:
    st.subheader("Kategori Nilai Ujian")
    cat = pd.cut(df.Exam_Score, [-np.inf, 60, 70, 80, 90, np.inf], right=False,
                 labels=["< 60", "60-69", "70-79", "80-89", "90-100"]).value_counts().sort_index()
    st.plotly_chart(style(go.Figure(go.Pie(labels=cat.index, values=cat.values, hole=.5,
        marker=dict(colors=[SUN, CAD, LIGHT, "#FFD9A8", BIS])))), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.subheader("Hubungan Dua Variabel")
    x = st.selectbox("X", NUM, index=NUM.index("Study_Hours"), format_func=LABELS.get)
    y = st.selectbox("Y", NUM, index=NUM.index("Exam_Score"), format_func=LABELS.get)
    fig = px.scatter(df, x=x, y=y, color="Gender", labels=LABELS,
                     color_discrete_map={"F": SUN, "M": CAD})
    if x != y:
        b, a0 = np.polyfit(df[x], df[y], 1)
        xs = np.array([df[x].min(), df[x].max()])
        fig.add_trace(go.Scatter(x=xs, y=a0 + b * xs, mode="lines", name="Regresi",
                                 line=dict(color=INK, dash="dash")))
        st.caption(f"r = {df[x].corr(df[y]):.2f}")
    st.plotly_chart(style(fig), use_container_width=True)
with c4:
    st.subheader("Rata-rata per Gender")
    m = df.groupby("Gender")[NUM].mean().T
    m = (m / m.max() * 100).rename(index=LABELS)  # skala relatif terhadap nilai terbesar
    fig = px.bar(m, barmode="group", orientation="h", color_discrete_map={"F": SUN, "M": CAD})
    st.plotly_chart(style(fig.update_layout(xaxis_title="% dari rata-rata tertinggi")),
                    use_container_width=True)

st.subheader("Matriks Korelasi (Pearson)")
cr = df[NUM].corr().rename(index=LABELS, columns=LABELS).round(2)
st.plotly_chart(style(px.imshow(cr, text_auto=True, zmin=-1, zmax=1,
    color_continuous_scale=[[0, SUN], [.5, BIS], [1, CAD]])), use_container_width=True)

st.subheader("Keputusan yang Bisa Diambil (data lengkap, n = 100)")
st.markdown("""
1. **Jadikan kehadiran alarm dini.** Kehadiran hampir identik dengan nilai (r = 0,99); 16 mahasiswa berkehadiran < 80% perlu dipantau.
2. **Prioritaskan kelompok berisiko.** 16 mahasiswa bernilai < 60 dan 22 berkepuasan <= 5. Nilai laki-laki 64,3 vs perempuan 83,3; usia 22 tahun 62,4 vs usia 18 tahun 86,8.
3. **Dorong belajar dan tidur cukup.** Belajar <= 8 jam rata-rata 59,2 vs > 12 jam 87,9; tidur <= 6 jam 60,2 vs > 7 jam 87,3.
4. **Kepuasan dan pengeluaran** adalah indikator pendukung, bukan target intervensi.
5. **Jangan simpulkan sebab-akibat.** Korelasi sangat tinggi dan gender bercampur dengan semua variabel; uji lanjutan (uji-t, regresi berganda) dan pilot program diperlukan.
""")
