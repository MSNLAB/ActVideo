import numpy as np
from sklearn.cluster import KMeans

from core.strategies.base_sampling import BaseSampling


class KMeansSampling(BaseSampling):

    def query(self, embeds, n):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(embeds)
        distance = kmeans.transform(embeds)
        query_ids = np.argmin(distance, axis=0)
        return query_ids
