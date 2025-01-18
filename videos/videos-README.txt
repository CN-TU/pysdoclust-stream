
These videos help to visualize and understand how SDOstreamclust processes data incrementally. Note that they have been created after the algorithm has processed the entire stream.

Each video has three plots:
(a) Predictions: SDOstreamclust clustering at time "t", showing points between [t-W,t+W]. 
(b) Observers/model: SDOstreamclust model (i.e., set of observers) at time "t".
(c) Ground Truth: the clustering provided by the Ground Truth at time "t", showing points between [t-W,t+W].

- Drift Conglomerates (W=100)
clustering.SDOstreamclust(k=500, T=600, outlier_handling=True, outlier_threshold=5, x=5) 

- Retail (W=15)
clustering.SDOstreamclust(k=75, T=150, outlier_handling=True, outlier_threshold=5, x=5, chi_prop=0.15, qv=0.1, e=3) 

- Fert vs Gdp (W=25)
clustering.SDOstreamclust(k=100, T=250, outlier_handling=True, outlier_threshold=5, x=5, chi_prop=2, qv=0.1, e=3) 

