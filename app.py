from flask import Flask, render_template, request, abort, jsonify
import pandas as pd
import numpy as np
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)

# =====================================
# LOAD DATASET
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset-wisata-jogja-sekitar.csv")

df = pd.read_csv(DATASET_PATH)

# =====================================
# DATA CLEANING
# =====================================
df = df.drop_duplicates(subset=['nama'], keep='first')
df = df.reset_index(drop=True)  # 🔑 index = ID wisata

df['type'] = df['type'].fillna('')
df['description'] = df['description'].fillna('')
df['image'] = df['image'].fillna('')
df['vote_average'] = df['vote_average'].fillna(0)
df['vote_count'] = df['vote_count'].fillna(0)
df['htm_weekday'] = pd.to_numeric(df['htm_weekday'], errors='coerce').fillna(10000)
df['htm_weekend'] = pd.to_numeric(df['htm_weekend'], errors='coerce').fillna(15000)

# =====================================
# DATA SIMULASI (UNTUK DASHBOARD)
# =====================================
df['pengunjung'] = (df['vote_count'] * 10).astype(int)
df['pendapatan'] = df['pengunjung'] * df['htm_weekday']

# =====================================
# TEXT CLEANING
# =====================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

df['content'] = (df['type'] + " " + df['description']).apply(clean_text)

# =====================================
# TF-IDF + DISTANCE
# =====================================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])
distance_matrix = euclidean_distances(tfidf_matrix)

# =====================================
# FUNGSI REKOMENDASI
# =====================================
def rekomendasi_wisata(nama, top_n=5):
    if nama not in df['nama'].values:
        return []

    idx = df.index[df['nama'] == nama][0]
    jarak = distance_matrix[idx]

    rekom_idx = np.argsort(jarak)[1:top_n+1]
    hasil = df.iloc[rekom_idx].copy()

    hasil['id'] = hasil.index  # 🔑 ID KONSISTEN

    return hasil.sort_values(
        by='vote_average',
        ascending=False
    ).to_dict(orient='records')

# =====================================
# ROUTE HOME / REKOMENDASI
# =====================================
@app.route("/", methods=["GET", "POST"])
def index():
    hasil = []
    tempat_dipilih = None

    if request.method == "POST":
        tempat_dipilih = request.form.get("tempat")
        hasil = rekomendasi_wisata(tempat_dipilih)

    return render_template(
        "index.html",
        tempat_list=df['nama'].tolist(),
        hasil=hasil,
        tempat_dipilih=tempat_dipilih
    )

# =====================================
# ROUTE DASHBOARD
# =====================================
@app.route("/dashboard")
def dashboard():
    data = df.copy()
    data['id'] = data.index

    total_destinasi = len(data)
    total_pengunjung = int(data['pengunjung'].sum())
    rata_pengunjung = round(total_pengunjung / 30, 2)
    total_pendapatan = int(data['pendapatan'].sum())
    rating_rata = round(data['vote_average'].mean(), 2)

    return render_template(
        "dashboard.html",
        wisata=data.to_dict(orient='records'),
        total_destinasi=total_destinasi,
        total_pengunjung=total_pengunjung,
        rata_pengunjung=rata_pengunjung,
        total_pendapatan=total_pendapatan,
        rating_rata=rating_rata
    )

# =====================================
# API FILTER (TANPA RELOAD)
# =====================================
@app.route("/api/wisata")
def api_wisata():
    kategori = request.args.get("kategori")
    rating = request.args.get("rating")
    harga = request.args.get("harga")
    sort = request.args.get("sort")  # rating_asc, rating_desc, harga_asc, harga_desc

    data = df.copy()

    if kategori and kategori != "all":
        data = data[data['type'] == kategori]

    if rating:
        data = data[data['vote_average'] >= float(rating)]

    if harga:
        data = data[data['htm_weekday'] <= int(harga)]

    if sort == "rating_asc":
        data = data.sort_values("vote_average", ascending=True)
    elif sort == "rating_desc":
        data = data.sort_values("vote_average", ascending=False)
    elif sort == "harga_asc":
        data = data.sort_values("htm_weekday", ascending=True)
    elif sort == "harga_desc":
        data = data.sort_values("htm_weekday", ascending=False)

    data['id'] = data.index

    return jsonify(data.to_dict(orient="records"))

# =====================================
# ROUTE DETAIL WISATA
# =====================================
@app.route("/wisata/<int:id>")
def detail(id):
    if id not in df.index:
        abort(404)

    wisata = df.loc[id]
    rekom = rekomendasi_wisata(wisata['nama'], 4)

    return render_template(
        "detail.html",
        w=wisata,
        rekom=rekom
    )


# =====================================
# RUN
# =====================================
if __name__ == "__main__":
    app.run(debug=True)
