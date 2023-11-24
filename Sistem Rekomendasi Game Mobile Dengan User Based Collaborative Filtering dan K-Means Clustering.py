from tkinter import *
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
#======================================================================================================================

# Input Data ratings dan games
ratings = pd.read_csv('rating.csv')
games = pd.read_csv('game.csv')

# Membentuk Pivot Table dari user_id, game_id, dan rating
pivotTable_ratings = ratings.pivot_table('rating', index='user_id', columns='game_id')

# Mengganti nilai Nan menjadi 0
pivotTableClustering = pivotTable_ratings.fillna(0)

# Scaling data menggunakan standardisasi
scl = StandardScaler()
standardisasiClustering = scl.fit_transform(pivotTableClustering)

# Reduksi Dimensi data menggunakan PCA
pca = PCA(n_components=2)
pcaClustering = pca.fit_transform(standardisasiClustering)
 
#======================================================================================================================

# Method Clustering
def clustering():
    global pvtTableCluster
    # Membuat Variabel Baru dengan nilai dari variabel pivotTableClustering
    pvtTableCluster = pivotTableClustering.copy()
    
    # Menginput nilai k
    k = int(selected_option_jumCluster.get())
    
    # Melakukan Clustering Menggunakan K-Means Clustering
    k_Means = KMeans(n_clusters = k, random_state=42, n_init=10)
    # fit model and predict clusters
    kMeansClustering = k_Means.fit_predict(pcaClustering)
    pvtTableCluster['Cluster'] = kMeansClustering
    
    # Memvisualisasikan Cluster
    # Mengatur besarnya  hasil visualisasi 
    fig = plt.figure(figsize=(5.94, 4))
    
    # Melabeli titik pada scatter plot
    plt.scatter(pcaClustering[:, 0], pcaClustering[:, 1], c=kMeansClustering, cmap='viridis', alpha=0.7, edgecolors='k')
    
    # Melabeli pusat-pusat kluster
    plt.scatter(k_Means.cluster_centers_[:, 0], k_Means.cluster_centers_[:, 1], marker='x', linewidths=2, zorder=10,
                s=200, c='red', label='Centroids')
    
    # Melabeli titik-titik dengan indeks kluster
    for i, label in enumerate(kMeansClustering):
        plt.annotate(label, (pcaClustering[i, 0], pcaClustering[i, 1]), textcoords="offset points", xytext=(0, 5), ha='center')
        
    plt.legend()
    
    # Melabeli sumbu X
    plt.xlabel('X')
    # Melabeli sumbu Y
    plt.ylabel('Y')
    
    
    # Membuat canvas untuk tampilan GUI
    canvas = FigureCanvasTkAgg(fig, master=frame_cluster)
    canvas.draw()
    canvas.get_tk_widget().place(x=5,y=70)
    
    # Menghitung Nilai Silhouette Coefficient
    sil_score = silhouette_score(pcaClustering, kMeansClustering)
    nilaiSC.set(round(sil_score,3))
    
    # Menghitung Jumlah user Tiap Cluster
    jmlCluster = pvtTableCluster['Cluster'].value_counts()
    
    # Membuat Treeview
    table_jumlahCluster = ttk.Treeview(frame_cluster)

    # Menambahkan kolom-kolom tabel
    table_jumlahCluster['columns'] = ('Cluster', 'Jumlah User')

    # Mengatur nama kolom
    table_jumlahCluster.column('#0', width=0, stretch=NO)
    table_jumlahCluster.column('Cluster', anchor=CENTER, width=210)
    table_jumlahCluster.column('Jumlah User', anchor=CENTER, width=210)

    # Membuat heading untuk setiap kolom
    table_jumlahCluster.heading('#0', text='', anchor=W)
    table_jumlahCluster.heading('Cluster', text='Cluster', anchor=CENTER)
    table_jumlahCluster.heading('Jumlah User', text='Jumlah User', anchor=CENTER)

    # Menambahkan data ke dalam tabel
    for index, row in jmlCluster.iteritems():
        table_jumlahCluster.insert(parent='', index='end', iid=str(index), text='', values=(index, row))

    # Mengatur tampilan tabel table_jumlahCluster
    table_jumlahCluster.place(x=5, y=450)

#======================================================================================================================

# Method Rekomendasi Game
def rekomendasi():
    # Membuat variabel baru dengan nilai dari pvtTableCluster
    pvtTableRekomen = pvtTableCluster.copy()
    
    # Menginput user_id
    userId = int(selected_option_user.get())
    
    # Menginput Neighbor
    nTetangga = int(selected_option_neighbor.get())
    
    # Menyimpan Rating asli dari user yang diinputkan
    pivotTableUser = pvtTableRekomen.copy()
    pivotTableUser = pivotTableUser.iloc[:,:-1]
    pivotTableUser = pd.DataFrame(pivotTableUser.loc[userId]).transpose().rename_axis(index=['user_id'])
    #print(pivotTableUser)
    
    # Merubah 20 Data Rating Menjadi 0 secara random
    pivotTableUserNew = pivotTableUser.stack()
    ratingnonzero = pivotTableUserNew[pivotTableUserNew != 0].sample(20, random_state=42)
    pivotTableUserNew.loc[ratingnonzero.index] = 0
    pivotTableUserNew = pivotTableUserNew.unstack()
    pivotTableUserNew = pivotTableUserNew.assign(Cluster=pvtTableRekomen.loc[userId, 'Cluster'])
    
    # Menyimpan Data Rating yang Sebelum Dirubah Menjadi 0
    ratingnonzerodf = ratingnonzero.to_frame().reset_index().rename(columns={'level_0': 'user_id', 0: 'rating'})
    ratingnonzeropt = ratingnonzerodf.pivot_table('rating', index=['user_id'], columns=['game_id'])
    
    barisPivot1 = pvtTableRekomen.loc[userId]
    barisPivot2 = pivotTableUserNew.loc[userId]
    pvtTableRekomen.loc[userId] = barisPivot2
    
    # Melihat cluster dari user yang akan diprediksi ratingnya
    userCluster = pvtTableRekomen.loc[userId]['Cluster']
    print(userCluster)
    
    # Menggangbungkan pivot tabel user dengan clusternya
    pivotTableCluster = pvtTableRekomen.copy()
    pivotTableCluster['Cluster'] = pvtTableRekomen['Cluster']
    
    # Mendapatkan timestamp sebelum eksekusi program
    start_time = time.time()
    
    # Hitung nilai similarity antar user menggunakan cosine similarity
    pivotTableClusterNew = pivotTableCluster[pivotTableCluster['Cluster'] == userCluster]
    pivotTableClusterNewDataFrame = pivotTableClusterNew.reset_index()
    X = pivotTableClusterNew.iloc[:, :-1]
    similarityMatrix = cosine_similarity(X)
    
    # Membuat Data Frame dari hasil hitung similarity menggunakan cosine similarity
    similarityMatrixDataFrame = pd.DataFrame(similarityMatrix, index= pivotTableClusterNewDataFrame['user_id'], columns=pivotTableClusterNewDataFrame['user_id'])
    
    # Prediksi Rating Menggunakan User-Based Collaborative Filtering
    def prediksiRating(userId, game_id, n):
      similarities = similarityMatrixDataFrame.loc[userId].drop(userId)
      similarities = similarities.nlargest(n)
      ratings = pvtTableRekomen.loc[similarities.index][game_id]
      weightedRating = ratings * similarities
      prediksi = weightedRating.sum() / similarities.sum()
      return prediksi
    
    # Membuat Variabel Baru menampung data rating yang sudah dirubah yang digunakan untuk melakukan rekomendasi
    pivotTableUserClusterHitungPred = pivotTableClusterNew.copy()
    pivotTable1UserModifHitungPred = pivotTableUserNew.copy()
    pivotTableUserClusterHitungPred = pivotTableUserClusterHitungPred.iloc[:,:-1]
    pivotTable1UserModifHitungPred = pivotTable1UserModifHitungPred.iloc[:,:-1]
    
    # Mengecek Jumlah Rating 0 Pada Data Rating User Yang Dipilih
    row = pivotTable1UserModifHitungPred.loc[userId]

    # Menghitung Prediksi Rating Untuk Data Rating 0 Pada User Yang Sudah Dipilih
    # Perulangan untuk mengakses nilai yang bernilai 0
    for game_id, value in row.items():
        if value == 0:
            predicted_rating = prediksiRating(userId, game_id, nTetangga)
            if (predicted_rating >= 0 and predicted_rating < 1):
                predicted_rating = 1
            pivotTableUserClusterHitungPred.loc[userId,game_id] = predicted_rating
    pivotTableUserClusterHitungPred = pivotTableUserClusterHitungPred.applymap(round)
    pivotTableUserClusterHitungPred = pivotTableUserClusterHitungPred.astype(np.float64)
    
    # Memilih kolom yang memiliki nilai awal 0
    hasilUserHitungPred = pd.DataFrame(pivotTableUserClusterHitungPred.loc[userId]).transpose().rename_axis(index=['user_id'])
    hasilPredRating = hasilUserHitungPred[(pivotTable1UserModifHitungPred == 0)]
    
    # Menampilkan hasil
    hasilPredRating = hasilPredRating.loc[userId].dropna()
    hasilPredRating = pd.DataFrame(hasilPredRating).transpose().rename_axis(index=['user_id'])
    
    # Mengubah 20 data rating asli dari variabel ratingAsli_pt menjadi bentuk data frame
    ratingAslidf = pd.melt(ratingnonzeropt.reset_index(), id_vars='user_id', value_vars=ratingnonzeropt.columns, var_name='game_id', value_name='rating')
    
    # Mengubah hasil rating prediksi dari variabel hasilPredRating menjadi bentuk data frame
    hasilPredRating_df = pd.melt(hasilPredRating.reset_index(), id_vars='user_id', value_vars=hasilPredRating.columns, var_name='game_id', value_name='rating')
    
    # Membandingkan nilai Rating berdasarkan kesamaan game_id
    resultGamePred = pd.merge(hasilPredRating_df, games, on='game_id', suffixes=('_df1', '_df2'))
    resultGamePred = resultGamePred.rename(columns={'rating': 'Rating_Prediksi'})
    resultGamePred = resultGamePred[['game_id', 'game_name', 'Rating_Prediksi']]
    
    # Rekomendasi Game berdasarkan prediksi rating tertinggi
    rekomendasi_game_df = resultGamePred.sort_values(by='Rating_Prediksi', ascending=False)
    
    # Mendapatkan timestamp setelah eksekusi program
    end_time = time.time()
    
    # Menghitung selisih waktu
    execution_time = end_time - start_time
    waktu_eksekusi.set(round(execution_time,4))
    
    # Membuat Treeview
    tableRekomendasiGame = ttk.Treeview(frame_rekomendasi_game)

    # Menambahkan kolom-kolom tabel sesuai dengan kolom pada data frame
    columns = list(rekomendasi_game_df.columns)
    tableRekomendasiGame['columns'] = columns
    
    # Mengatur tinggi baris
    tableRekomendasiGame.configure(height=25)

    # Mengatur nama kolom
    for column in columns:
        tableRekomendasiGame.column(column, anchor=CENTER, width=100)
        tableRekomendasiGame.heading(column, text=column, anchor=CENTER)
    
    tableRekomendasiGame.column('#0', width=0, stretch=NO)
    tableRekomendasiGame.column('game_id', width=60, stretch=NO)
    tableRekomendasiGame.column('game_name', width=222, stretch=NO)
    
    tableRekomendasiGame.heading('#0', text='', anchor=W)

    # Menampilkan data
    for index, row in rekomendasi_game_df.iterrows():
        tableRekomendasiGame.insert(parent='', index='end', iid=str(index), values=list(row))

    # Mengatur tampilan tabel tableRekomendasiGame
    tableRekomendasiGame.place(x=5, y=85)        

#======================================================================================================================
    
    # Menghitung Nilai MAE tiap prediksi dan Rata-rata Mae pada 15 rating prediksi  
    # Mengganti nama kolom
    result = pd.merge(hasilPredRating_df, ratingAslidf, on='game_id', suffixes=('_df1', '_df2'))
    result = result.rename(columns={'user_id_df1': 'user_id', 'rating_df1': 'Rating_Prediksi', 'rating_df2': 'Rating_Asli'})
    
    # Menampilkan hasil perbandingan
    result = result[['user_id', 'game_id', 'Rating_Prediksi', 'Rating_Asli']]
    resultGameMae = pd.merge(result, games, on='game_id', suffixes=('_df1', '_df2'))
    resultGameMae
    
    # Mengganti nama kolom
    resultGameMae = resultGameMae.rename(columns={'game_name': 'judul_game',})
    
    # Menggabungkan kolom rating asli dengan kolom rating prediksi
    resultGameMae = resultGameMae[['game_id', 'judul_game', 'Rating_Prediksi', 'Rating_Asli']].sort_values(by='Rating_Prediksi', ascending=False)
    
    # Menambahkan kolom baru 'MAE' dan mengisi nilai dengan selisih rating asli dan rating prediksi
    mae = resultGameMae.assign(MAE = abs(resultGameMae['Rating_Asli'] - resultGameMae['Rating_Prediksi']))
    
    # Menghitung rata-rata dari kolom 'MAE'
    maeFinal = mae['MAE'].mean()
    nilaiMAE.set(round(maeFinal,3))
    
    # Membuat Treeview
    tableMae = ttk.Treeview(frame_mae)

    # Menambahkan kolom-kolom tabel sesuai dengan kolom pada data frame
    columns = list(mae.columns)
    tableMae['columns'] = columns
    
    # Mengatur tinggi baris
    tableMae.configure(height=20)

    # Mengatur nama kolom
    for column in columns:
        tableMae.column(column, anchor=CENTER, width=100)
        tableMae.heading(column, text=column, anchor=CENTER)
    
    tableMae.column('#0', width=0, stretch=NO)
    tableMae.column('game_id', width=55, stretch=NO)
    tableMae.column('judul_game', width=214, stretch=NO)
    tableMae.column('Rating_Asli', width=75, stretch=NO)
    tableMae.column('Rating_Prediksi', width=90, stretch=NO)
    tableMae.column('MAE', width=35, stretch=NO)
    
    tableMae.heading('#0', text='', anchor=W)

    # Menampilkan data
    for index, row in mae.iterrows():
        tableMae.insert(parent='', index='end', iid=str(index), values=list(row))

    # Mengatur tampilan tabel tableMae
    tableMae.place(x=2, y=20)
    
#======================================================================================================================

window = Tk()

window.title('Sistem Rekomendasi Game Mobile Dengan User Based Collaborative Filtering dan K-Means Clustering')

window.state("zoomed")

window.config(bg="#62ABC1")

#Title GUI
label_window = Label(window, text="SISTEM REKOMENDASI GAME MOBILE", font=("Times New Roman", 18, "bold"), bg="#62ABC1")
label_window.grid(row=0, column=0, columnspan=3, padx=0, pady=0)

#======================================================================================================================

#Frame K-Means Clustering
frame_cluster = LabelFrame(window, text="K-Means Clustering", font=("Times New Roman", 12, "bold"), bg="#55CFC8", width=440, height=705)
frame_cluster.grid(row=1, column=0, padx=7, pady=5)

# Option Menu Pilih Jumlah Cluster
# Label Title Cluster
pilih_cluster = Label(frame_cluster, text="Cluster", font=("Times New Roman", 12 , "bold"), bg="#55CFC8")
pilih_cluster.place(x=15, y=15)

# Daftar opsi Cluster
options_jumCluster = ["{}".format(i) for i in range(2, 11)]
# Variabel untuk menyimpan opsi yang dipilih
selected_option_jumCluster = StringVar(frame_cluster)
# Set nilai awal
selected_option_jumCluster.set(options_jumCluster[0])
# Membuat OptionMenu
option_menu_jumCluster = OptionMenu(frame_cluster, selected_option_jumCluster, *options_jumCluster)
option_menu_jumCluster.config(width=15)
option_menu_jumCluster.place(x=100, y=14)

#Button Clustering
button_cluster = Button(frame_cluster, text="Clustering", width=14, font=("Times New Roman", 12, "bold"), bg="#62ABC1", command=clustering)
button_cluster.place(x=280, y=14)

#Title Visualisasi Cluster
visual_cluster = Label(frame_cluster, text="Visualisasi Cluster", font=("Times New Roman", 10, "bold"), bg="#62ABC1")
visual_cluster.place(x=155, y=50)

#Title silhouette coefficient
silhouette_coefficient = Label(frame_cluster, text="Silhouette Coefficient", font=("Times New Roman", 10, "bold"), bg="#62ABC1")
silhouette_coefficient.place(x=153, y=365)

#Nilai silhouette coefficient
nilaiSC = StringVar()
nilaiSC.set('---')
nilai_silhouette = Label(frame_cluster, textvariable=nilaiSC, font=("Times New Roman", 16, "bold"), bg="#62ABC1")
nilai_silhouette.place(x=190, y=387)

#Title Jumlah user tiap cluster Cluster
jum_cluster = Label(frame_cluster, text="Jumlah Cluster", font=("Times New Roman", 10, "bold"), bg="#62ABC1")
jum_cluster.place(x=169, y=430)

#======================================================================================================================

#Frame Rekomendasi Game
frame_rekomendasi_game = LabelFrame(window, text="Rekomendasi Game", font=("Times New Roman", 12, "bold"), bg="#55CFC8", width=400, height=705)
frame_rekomendasi_game.grid(row=1, column=1, padx=7, pady=5)

# Option Menu Pilih User
# Label Title User
pilih_cluster = Label(frame_rekomendasi_game, text="User_ID", font=("Times New Roman", 12, "bold"), bg="#55CFC8")
pilih_cluster.place(x=15, y=15)

# Daftar opsi User
options_user = ["{}".format(i) for i in range(1, 21)]
# Variabel untuk menyimpan opsi yang dipilih
selected_option_user = StringVar(frame_rekomendasi_game)
# Set nilai awal
selected_option_user.set(options_user[0])
# Membuat OptionMenu
option_menu_user = OptionMenu(frame_rekomendasi_game, selected_option_user, *options_user)
option_menu_user.config(width=15)
option_menu_user.place(x=100, y=14)

#----------------------------------------------------------------------------------------------------------------------

# Label Title Neighbor
pilih_neighbor = Label(frame_rekomendasi_game, text="Neighbor", font=("Times New Roman", 12, "bold"), bg="#55CFC8")
pilih_neighbor.place(x=15, y=50)

# Daftar opsi Neighbor
options_neighbor = ["{}".format(i) for i in range(1, 21)]
# Variabel untuk menyimpan opsi yang dipilih
selected_option_neighbor = StringVar(frame_rekomendasi_game)
# Set nilai awal
selected_option_neighbor.set(options_neighbor[0])
# Membuat OptionMenu
option_menu_neighbor = OptionMenu(frame_rekomendasi_game, selected_option_neighbor, *options_neighbor)
option_menu_neighbor.config(width=15)
option_menu_neighbor.place(x=100, y=50)

#Button Rekomendasi
button_rekomendasi_game = Button(frame_rekomendasi_game, text="Rekomendasi", width=14, font=("Times New Roman", 12, "bold"), bg="#62ABC1", command=rekomendasi)
button_rekomendasi_game.place(x=250, y=28)

#Title WAKTU EKSEKUSI
title_waktu = Label(frame_rekomendasi_game, text="WAKTU EKSEKUSI :", font=("Times New Roman", 10, "bold"), bg="#62ABC1")
title_waktu.place(x=50, y=650)

#Nilai WAKTU EKSEKUSI
waktu_eksekusi = StringVar()
waktu_eksekusi.set('---')
waktu_detik = Label(frame_rekomendasi_game, textvariable=waktu_eksekusi, font=("Times New Roman", 14, "bold"), bg="#62ABC1")
waktu_detik.place(x=225, y=645)

detik = Label(frame_rekomendasi_game, text="Detik", font=("Times New Roman", 14, "bold"), bg="#62ABC1")
detik.place(x=315, y=645)

#======================================================================================================================

#Frame MAE
frame_mae = LabelFrame(window, text="MAE", font=("Times New Roman", 12, "bold"), bg="#55CFC8", width=480, height=705)
frame_mae.grid(row=1, column=2, padx=7, pady=5)

#Title MAE
title_mae = Label(frame_mae, text="NILAI MAE", font=("Times New Roman", 10, "bold"), bg="#62ABC1")
title_mae.place(x=200, y=1)

#Title rata-rata MAE
title_mae = Label(frame_mae, text="NILAI RATA-RATA MAE", font=("Times New Roman", 10, "bold"), bg="#62ABC1")
title_mae.place(x=175, y=500)

#Nilai Rata-rata MAE
nilaiMAE = StringVar()
nilaiMAE.set('---')
nilai_Mae = Label(frame_mae, textvariable=nilaiMAE, font=("Times New Roman", 16, "bold"), bg="#62ABC1")
nilai_Mae.place(x=225, y=525)

#======================================================================================================================
window.mainloop()