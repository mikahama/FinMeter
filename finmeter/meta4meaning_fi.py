from .utils.metaphor import *
from sklearn import preprocessing
from mikatools import *


class Meta4meaningFi:
    def __init__(self, rows_path, matrix_path):
        # get the rows and their indecies
        self.rows = load_termidx(rows_path)
        self.rev_rows = {i: r for r, i in self.rows.items()}
        sorted_rows = sorted(self.rows.items(), key=lambda k: k[1])
        self.sorted_rows = list(map(lambda r: r[0], sorted_rows))

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
            row = zip(self.sorted_rows, row)  # word: relatedness_score
            if positive:
                row = filter(lambda r: r[1] > 0, row)  # remove non-related words
            row = dict(row)
            return row
        except Exception as e:
            return None

    def get_sorted_rel(self, word, normalize=True, positive=True):
        row = self.get_rel(word, normalize, positive)
        if row:
            row = sorted(row.items(), key=lambda k: k[1], reverse=True)
        return row

    def interpret(self, tenor, vehicle):
        tv = self.get_rel(tenor)
        vv = self.get_rel(vehicle)

        if not tv or not vv:
            return []

        features = list(set(tv.keys()) | set(vv.keys()))
        # consider only concrete features
        # features = list(filter(lambda f: is_concrete(f), features))

        mv = [tv[f] * vv[f] if f in tv and f in vv else 0.0 for f in features]  # multiplication
        odv = [vv[f] - tv[f] if f in tv and f in vv and tv[f] > 0 and vv[f] > 0 else float('-inf') for f in
               features]  # overlap difference

        # map to features
        mvf = zip(features, mv)
        odvf = zip(features, odv)

        # sort them and convert back into ordered features
        mvf = sorted(mvf, key=lambda k: k[1], reverse=True)
        odvf = sorted(odvf, key=lambda k: k[1], reverse=True)
        mvf = list(map(lambda r: r[0], mvf))
        odvf = list(map(lambda r: r[0], odvf))

        # ranked interpretations
        interpretations = map(lambda f: (f, min([mvf.index(f), odvf.index(f)])), features)
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
