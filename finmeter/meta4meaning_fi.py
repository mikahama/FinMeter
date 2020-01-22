from .utils.metaphor import *
from sklearn import preprocessing
from mikatools import *


class Meta4meaningFi:
    def __init__(self, rows_path, matrix_path):
        # get the rows and their indecies
        self.rows = load_termidx(rows_path)

        cols_path = rows_path  # They are the same in our case

        # get the columns and their indecies
        self.cols = load_termidx(cols_path)

        self.rev_cols = {i: r for r, i in self.cols.items()}
        sorted_cols = sorted(self.cols.items(), key=lambda k: k[1])
        self.sorted_cols = list(map(lambda r: r[0], sorted_cols))

        # load the matrix
        self.matrix = hkl.load(matrix_path)

    def get_vector(self, word, normalize=True):
        try:
            i = self.rows[word]  # get the index of the word in the matrix
            row = self.matrix[i, :].A[0]  # get it's relatedness vector/row
            if normalize:
                row = preprocessing.normalize(row[:, np.newaxis], axis=0, norm='l1').ravel()
            return row
        except Exception as e:
            return None

    def get_rel(self, word, normalize=True, positive=True):
        try:
            i = self.rows[word]  # get the index of the word in the matrix
            row = self.matrix[i, :].A[0]  # get it's relatedness vector/row
            if normalize:
                row = preprocessing.normalize(row[:, np.newaxis], axis=0, norm='l1').ravel()
            row = zip(self.sorted_cols, row)  # word: relatedness_score
            if positive:
                row = filter(lambda r: r[1] > 0, row)  # remove non-related words
            row = dict(row)
            return row
        except Exception as e:
            return None

    def get_sorted_rel(self, word, normalize=True, positive=True, k=0):
        row = self.get_rel(word, normalize, positive)
        if row:
            row = sorted(row.items(), key=lambda k: k[1], reverse=True)
            if k > 0 and k < len(row):
                row = row[:k]
        return row

    def interpret(self, tenor, vehicle):
        tv = self.get_vector(tenor)
        vv = self.get_vector(vehicle)

        if tv is None or vv is None:
            return []

        features = np.union1d(np.where(tv > 0), np.where(vv > 0))  # all non-zero features
        shared_features = np.intersect1d(np.where(tv > 0), np.where(vv > 0))  # shared features

        # consider only concrete features? add filter here

        mv = np.zeros(tv.shape)
        mv[shared_features] = tv[shared_features] * vv[shared_features]  # multiplication
        mv = mv[features]

        odv = np.zeros(tv.shape)
        odv[features] = float('-inf')
        odv[shared_features] = vv[shared_features] - tv[shared_features]  # overlap difference
        odv = odv[features]

        # sort them and convert back into ordered features
        mvf = np.argsort(-mv)
        odvf = np.argsort(-odv)

        mvf_l = mvf.tolist()
        odvf_l = odvf.tolist()

        # ranked interpretations
        interpretations = [(self.rev_cols[features[f]], min([mvf_l.index(f), odvf_l.index(f)])) for f in
                           range(len(features))]
        return list(sorted(interpretations, key=lambda k: k[1]))

    def metaphoricity(self, tenor, vehicle, expression, k=0):
        tv = self.get_sorted_rel(tenor, normalize=False)
        vv = self.get_sorted_rel(vehicle, normalize=False)

        if not tv or not vv:
            return 0.0

        if k == 0:
            pass
        elif k > 0 and k < len(self.rows):
            tv = tv[:k]
            vv = vv[:k]
        else:
            raise Exception("k must be positive and less than %s" % len(self.rows))

        tv = dict(tv)
        vv = dict(vv)

        t_relatedness = np.max([tv.get(t, 0.0) for t in expression])  # relatedness to tenor
        v_relatedness = np.max([vv.get(t, 0.0) for t in expression])  # relatedness to vehicle
        tv_score = t_relatedness * v_relatedness

        vt_diff = np.max([vv.get(t, 0.0) - tv.get(t, 0.0) for t in expression])  # vehicle tenor difference

        # negative difference is the maximum, i.e. words related to tenor more than the vehicle
        # return (tv_score, vt_diff,) if tv_score > 0 and vt_diff > 0 else (0.0, 0.0,)
        return np.mean([tv_score, vt_diff]) if tv_score > 0 and vt_diff > 0 else 0.0


def main():
    rows_path = script_path('data/metaphor/unigrams_sorted_5k.txt')
    matrix_path = script_path('data/metaphor/rel_matrix_n_csr.hkl')
    m4m = Meta4meaningFi(rows_path=rows_path, matrix_path=matrix_path)
    pprint(m4m.get_sorted_rel('koira')[:30])
    print()

    pprint(m4m.interpret('mies', 'lamppu')[:30])
    print()
    pprint(m4m.interpret('asianajaja', 'hai')[:30])
    print()

    # 0 to select all
    pprint(m4m.metaphoricity('lintu', 'rakkaus', ['äiti', 'maa', 'seksi', 'luonto', 'suloinen'], 300))


if __name__ == '__main__':
    main()
