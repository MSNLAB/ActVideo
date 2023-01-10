class BaseExampler:

    def __init__(self, maxsize=4096, *args, **kwargs):
        self.maxsize = maxsize
        self.caches = []

    @property
    def rest_size(self):
        return self.maxsize - self.used_size

    @property
    def used_size(self):
        return len(self.caches)

    def insertable(self, insert_cnt=0):
        return self.rest_size >= insert_cnt

    def insert(self, data, *args, **kwargs):
        if not isinstance(data, (list, tuple)):
            data = (data,)

        insert_len = len(data)
        assert self.insertable(insert_len), \
            f'cache can not insert more data, ' \
            f'current space [{self.rest_size}/{self.maxsize}], ' \
            f'insert needed {insert_len}'

        self.caches.extend(data)
        return data

    def reduce(self, ids, *args, **kwargs):
        assert set(ids) <= set(list(range(self.used_size))), \
            f'reduce cache ids are out of memory.'
        self.caches = self.replay(
            list(set(list(range(self.used_size))).difference(ids)))
        return ids

    def replay(self, ids, *args, **kwargs):
        assert set(ids) <= set(list(range(self.used_size))), \
            f'replay cache ids are out of memory.'
        return [self.caches[idx] for idx in ids]
