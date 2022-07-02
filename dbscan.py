import json , io, base64
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
from sklearn import preprocessing
from sklearn import cluster , metrics
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from db import con
# conn= psycopg2.connect(database="giinwedb", user="modulo4", password="modulo4", host="128.199.1.222", port="5432")
SILHOUETTE_SAMPLE_SIZE=3000

def get_dataFrame(sql, con):
    df = pd.read_sql_query(sql, con = con)
    #print(df)
    return df

def transform_data(dat):
    label_encoder = preprocessing.LabelEncoder()
    transformed_data = dat.apply(label_encoder.fit_transform)
    return transformed_data

def dbscan_model(eps, min_samples, query):
    result = {}

    #TODO: Obtener data desde la query
    data=get_dataFrame(query, con)
    #TODO: transformamos la data
    dataTransformed = transform_data(data)   
    # inicializamos DBSCAN
    clustering_model=DBSCAN(eps=eps,min_samples=min_samples)
    # ajustamos el modelo a transform_data
    clustering_model.fit_predict(dataTransformed)
    predicted_labels=clustering_model.labels_    
    
    data['cluster'] = predicted_labels
    #print(data['cluster'])    

    ########metrics and number of clusters####################
    n_clusters_ = len(set(predicted_labels)) - (1 if -1 in predicted_labels else 0)
    n_noise_    = list(predicted_labels).count(-1)    
    
    #2 <= n_labels <= n_samples - 1
    if((len(set(predicted_labels))> 1) and (len(set(predicted_labels))<= (min_samples - 1))):        
        coefficient = metrics.silhouette_score(dataTransformed, clustering_model.labels_ , sample_size=SILHOUETTE_SAMPLE_SIZE)
    else:
        coefficient = -1
    
    clusters_uniques = set(list(predicted_labels))
    cant = list(predicted_labels)    
    
    metricas_totales = []
    cantidad_cluster = {}
    n_noise_porcentaje= 0.0
    for item in clusters_uniques:        
        if(len(cant) != 0):
             if(item != -1):
                cantidad_cluster = {
                    "clusters": int(item),
                    "cantidad": cant.count(int(item)),
                    "porcentaje": "{:.5f}".format(float(cant.count(int(item))/len(cant)))    
                        }
                metricas_totales.append(cantidad_cluster)
             else:
                 n_noise_porcentaje="{:.5f}".format(float(cant.count(int(item))/len(cant)))
        else:
            cantidad_cluster = {
                "clusters": int(item),
                "cantidad": cant.count(int(item)),
                "porcentaje": 0.0   
                    }
            metricas_totales.append(cantidad_cluster)

    #result['data'] = tuplas
    #console.log(tuplas)
    #print(json.loads(data.to_json(orient = 'records')))

    result['data'] = json.loads(data.to_json(orient = 'values'))#'values'
    #result['data'] = list(data.values)
    #print(result['data'])

    result['metricas'] = { 
                'n_clusters': n_clusters_,
                'n_noise': n_noise_,
                'Coefficient': "{:.5f}".format(coefficient),
                'n_noise_porcentaje': n_noise_porcentaje
                 }

    ##############visualizacion de DBSCAN ##################
    #visualzing clusters

    dataTransformed['cluster']=predicted_labels
    
    clusters = dataTransformed['cluster'].apply(lambda x: 'cluster ' +str(x) if x != -1 else 'outlier')
    numero_clusters= len(set(clusters))
    ##print(numero_clusters)
    XX=dataTransformed.iloc[:,[0,1]].values

    plt.figure(figsize=(13,10))
            
    for i in range(numero_clusters):
        if (i-1) != -1:
            plt.scatter(XX[predicted_labels== (i-1), 0], XX[predicted_labels==(i-1), 1], s=80, cmap='Paired', label = clusters.unique())
        else:
            plt.scatter(XX[predicted_labels== (i-1), 0], XX[predicted_labels==(i-1), 1], s=50, c='Grey', label = clusters.unique())           


    plt.legend(clusters.unique(),bbox_to_anchor=(0.99,1),fontsize=12)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.6)
    plt.xlabel('Categoria')
    plt.ylabel('Datos')
    plt.title("DBSCAN")
    #plt.show()
 
    ##############
    
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    # plt.savefig("graphic2.jpg")
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
    result["graphic_dbscan"] = my_base64_jpgData.decode()

    plt.figure(clear=True) 
    result["graphic_method_codo"] =''#show_codo(dataTransformed)

    result["numColumn"] = list(data.columns.values)
    #print("cabecera", list(data.columns.values))
    result["metricas_detalles"] = metricas_totales
    #print("colum", result['numColumn'])
    #print(result['data'])

    return result





