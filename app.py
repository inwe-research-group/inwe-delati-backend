# IMPORTS
#import os, psycopg2, json, io, base64
import json, io, base64
import pandas as pd
# LIB
from scipy import spatial
from sklearn import preprocessing,metrics
# FLASK 
from flask import Flask, request, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
# maching learning
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm 
import numpy as np
#
from decouple import config
from libreria import *
from db import con
app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
#app.config.from_object(os.environ['APP_SETTINGS'])
#app.config['SQLALCHEMY_DATABASE_URI']='postgresql://128.199.1.222/giinwedb'
#app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

PORT=config('PORT_BACK')
HOST=config('HOST')
DEBUG=config('DEBUG')


db = SQLAlchemy(app)

def load_data():      
    #Obtiene la data en formato data frame
    result=get_dataFrame("select distinct o.htitulo_cat, o.htitulo, w.pagina_web, o.empresa, o.lugar, o.salario, date_part('year',o.fecha_publicacion) as periodo, f_dimPuestoEmpleo(o.id_oferta,7) as funciones, f_dimPuestoEmpleo(o.id_oferta,1) as conocimiento, f_dimPuestoEmpleo(o.id_oferta,3) as habilidades, f_dimPuestoEmpleo(o.id_oferta,2) as competencias, f_dimPuestoEmpleo(o.id_oferta,17) as certificaciones, f_dimPuestoEmpleo(o.id_oferta,5) as beneficio, f_dimPuestoEmpleo(o.id_oferta,11) as formacion from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) where o.id_estado is null;", con)
    #Convierte a un formato json
    result=json.loads(result.to_json(orient = 'values'))
    return result

@app.route("/algorithms", methods = ['GET', 'POST'])
def algorithms():
    #Algoritmos que se pueden usar
    name_algorithms = ['kmeans','MetododelCodo','Clasificacion','DBSCAN'] #, + algorithms    
    return jsonify({'algorithms':name_algorithms}) 

@app.route("/MetododelCodo", methods = ['GET', 'POST'])
def MetododelCodo():      
    if request.method == 'GET':
        return jsonify(load_data())
    if request.method == 'POST':
        body        = request.get_json()      
        n_clustersMax  = body["n_clusters"]
        init        = body['init']
        n_init      = body['n_init']
        random_state= body['random_state']
        max_iter    = body['max_iter']        
        result      = {}
        # end requests+
        #Obtenemos la data
        dataframe = get_dataFrame(body["query"], con)
        #Transformamos la data
        transformed_data = transform_dataFrame(dataframe)
        #Normalizamos la data
        transformed_data= normalization_dataFrame(transformed_data)
        #metodo del codo          
        distortions = []
        #Cantidad de clusters de 2 hasta n
        K = range(2,n_clustersMax+1)
        #Hallamos la distorcion para cada cluster
        for k in K:
            kmeanModel = KMeans(n_clusters=k, init=init, max_iter=max_iter, n_init=n_init, random_state=random_state)
            kmeanModel.fit(transformed_data)
            distortions.append(kmeanModel.inertia_)
        #Dibujamos la grafica
        plt.plot(K, distortions, 'x-b')
        plt.xlabel('k clusters')
        plt.ylabel('Distorción')
        plt.title('El método del codo muestra el k clusters óptimo')
        #Grafica a formato base64
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')        
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
        result["elbow_method"] = my_base64_jpgData.decode()
        plt.clf() #clear current image plt 
        response = jsonify(result)
        return response        
       
@app.route("/Clasificacion", methods = ['GET', 'POST'])
def Clasificacion():     
    if request.method == 'GET':
        return jsonify(load_data())
    if request.method == 'POST':
        body        = request.get_json()            
        n_clusters  = body["n_clusters"]
        init        = body['init']
        n_init      = body['n_init']
        random_state= body['random_state']
        max_iter    = body['max_iter']        
        axis_x      = int(body['axis_x'])
        axis_y      = int(body['axis_y'])
        result      = {}
        # end requests+
        #Obtenemos la data
        dataframe = get_dataFrame(body["query"], con)
        #Transformamos la data
        transformed_data = transform_dataFrame(dataframe)
        #Normalizamos la data
        transformed_data=normalization_dataFrame(transformed_data)
        #Nombres de las columnas
        field_names = list(transformed_data.columns.values)
        # KMEANS
        kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init, random_state=random_state)        
        kmeans.fit_predict(transformed_data)
        #Clusters de cada fila
        elements = kmeans.labels_  # values from kmeans.fit_predict(transformed_data)
        #Centros de los clusters
        centroids = kmeans.cluster_centers_
        #Hallamos el elementos mas cercano al centroides
        centroids_values = [] # for search in dataframe
        centroids_all_data = [] # centroids details
        tree = spatial.KDTree(transformed_data)
        for cd in centroids:            
            found = tree.query(cd)
            centroids_values.append(found[1])
            centroids_all_data.append(found)    
        #Agregamos la columna cluster a la data
        dataframe["cluster"] = elements
        #Ordenamos la data en base al cluster
        dataframe.sort_values(['cluster'], ascending=False)    
        #Agregamos a los nombres de las columnas cluster    
        field_names.append("cluster")
        #Hallamos los datos para cada centroide
        centroids_details = []
        x = 0
        for _centroid in centroids_all_data:
            obj = {}
            obj["point"] = (centroids.tolist())[x]
            obj["distance"] = float(_centroid[0])
            obj["position"] = int(_centroid[1])
            obj["title_cluster"]= json.loads((dataframe.iloc[centroids_values[x]]).to_json(orient='values'))
            centroids_details.append(obj)
            x+=1
        #Agregamos la data al json
        result["centroids"] = centroids_details
        result["inertia"] = kmeans.inertia_
        result["n_iter"] = kmeans.n_iter_
        result["total_instances"] = len(dataframe.index)
        result["columns"] = field_names
        result["data"] = json.loads(dataframe.sort_values(['cluster'], ascending=True).to_json(orient='table'))
        #Hallamos los datos de los clusters
        clusters = []
        nombreClusters=[]
        cantidadClusters=[]
        porcentajes=dataframe["cluster"].value_counts(normalize=True)
        size_clusters=dataframe["cluster"].value_counts()
        for item in range(n_clusters):
            temporal_cluster = 'Cluster {}'.format(item)
            length_actual_cluster = int(size_clusters[item])            
            nombreClusters.append(temporal_cluster)
            cantidadClusters.append(length_actual_cluster)
            obj = {
                "cluster": temporal_cluster,
                "length": length_actual_cluster,
                "percentage": (round(float(porcentajes[item])*100, 2)),
                "title_cluster": json.loads((dataframe.iloc[centroids_values[item]]).to_json(orient='values'))                
            }
            clusters.append(obj)
        #Agregamos los datos del cluster al json
        result["clusters"] = clusters           
        result["elbow_method"] = ''
        #============================
        #2 componentes ACP
        result["2CP"]=acp(transformed_data,kmeans,2)
        #3 componentes ACP
        result["3CP"]=acp(transformed_data,kmeans,3)
        #Generamos imagen 2D
        result["graphic_kmeans_2D"]=imagen2D( result["2CP"],'KMEANS')
        #Generamos imagen 3D
        result["graphic_kmeans_3D"]=imagen3D( result["3CP"],'KMEANS')
        #Cambiamos DataFrame 2CP a json
        result["2CP"]=json.loads(result["2CP"].to_json(orient='values'))
        #Cambiamos DataFrame 3CP a json
        result["3CP"]=json.loads(result["3CP"].to_json(orient='values'))
        result["graphic"] = result["graphic_kmeans_2D"]
        response = jsonify(result)
        return response
    
@app.route('/dbscan', methods=['POST'])
def dbscan ():    
    if request.method == 'POST':
        body = request.get_json()       
        query       = body['query']
        eps         = body['eps']
        min_samples = body['min_samples']     
        total_data={}
        #Obtener data desde la query
        data=get_dataFrame(query, con)
        #Transformamos la data
        dataTransformed = transform_dataFrame(data)  
        #Normalizar la data
        normdata=normalization_dataFrame(dataTransformed)
        #DBSCAN
        clustering_model=DBSCAN(eps,min_samples=min_samples)
        #Ingresamos data normalizada
        clustering_model.fit_predict(normdata)
        #Clusters encontrados
        predicted_labels=clustering_model.labels_
        #Silhouette_score
        if((len(set(predicted_labels))- (1 if -1 in predicted_labels else 0)> 1) and (len(set(predicted_labels))<= (min_samples - 1))):        
            coefficient = metrics.silhouette_score(dataTransformed.iloc[predicted_labels!=-1], [a for a in predicted_labels if a!=-1])
        else:
            coefficient = -1     
        #Variables
        clusters_uniques = set(list(predicted_labels))
        cant = list(predicted_labels)    
        cores = [predicted_labels[i] for i in clustering_model.core_sample_indices_ ]
        cluster_detalles = []
        cantidad_cluster = {}
        n_noise_porcentaje= 0.0
        #Para cada cluster
        for item in clusters_uniques:        
            if(item != -1):
                #Identificador de cluster
                #Cantidad de elementos
                #Porcentaje de elementos comparado con el total
                cantidad_cluster = {
                        "cluster": int(item),
                        "nucleos":  cores.count(int(item)),
                        "cantidad": cant.count(int(item)),
                        "porcentaje": "{:.5f}".format(float(cant.count(int(item))/len(cant))*100)    
                            }
                #Agregar a cluster_detalles
                cluster_detalles.append(cantidad_cluster)
            else:
                #Porcentaje de ruido
                n_noise_porcentaje="{:.5f}".format(float(cant.count(int(item))/len(cant))*100)
        #Metricas obtenidas
        total_data["cluster_detalles"] = cluster_detalles
        #Columnas de la data
        total_data["columns"] = list(data.columns.values)
        #Data del query con columna cluster
        data["cluster"]=clustering_model.labels_
        data["idx"]=[i for i in range(len(list(clustering_model.labels_)))]
        total_data["data"]=json.loads(data.to_json(orient='values'))
        #Indices de los core
        total_data["dbscan_core_indices"] = clustering_model.core_sample_indices_.tolist()
        #Metricas
        total_data["dbscan_metricas"]={
                'dbscan_clusters': len(set(clustering_model.labels_)) - (1 if -1 in clustering_model.labels_ else 0),
                'dbscan_noise': list(clustering_model.labels_).count(-1),
                'dbscan_Coefficient': "{:.5f}".format(coefficient),
                'dbscan_noise_porcentaje': n_noise_porcentaje,
        }
        #2 componentes ACP
        total_data["dbscan_2CP"]=acp(normdata,clustering_model,2)
        #3 componentes ACP
        total_data["dbscan_3CP"]=acp(normdata,clustering_model,3)
        #Generamos imagen 2D
        total_data["graphic_dbscan_2D"]=imagen2D( total_data["dbscan_2CP"],'DBSCAN')
        #Generamos imagen 3D
        total_data["graphic_dbscan_3D"]=imagen3D( total_data["dbscan_3CP"],'DBSCAN')
        #Cambiamos DataFrame 2CP a json
        total_data["dbscan_2CP"]=json.loads(total_data["dbscan_2CP"].to_json(orient='values'))
        #Cambiamos DataFrame 3CP a json
        total_data["dbscan_3CP"]=json.loads(total_data["dbscan_3CP"].to_json(orient='values'))
        return (total_data)

if __name__ == '__main__':
    #app.run()
    #app.run(debug=True)
    #app.run(port=PORT, debug=DEBUG)
    app.run(host=HOST,port=PORT,debug=DEBUG)
    