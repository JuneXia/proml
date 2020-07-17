import numpy as np
import os
from sklearn.cluster import KMeans


n_clusters = 5


centers = {}
for label in range(opt.label_nc):
    n_clusters = min(feat.shape[0], opt.n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feat)
    centers[label] = kmeans.cluster_centers_
save_name = os.path.join(save_path, name + '_clustered_%03d.npy' % opt.n_clusters)
np.save(save_name, centers)
print('saving to %s' % save_name)