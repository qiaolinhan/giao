'''
Using IOU matching to make linear assignment
----------
Parameters
----------
cost: ndarray
An N * M matrix for costs between each tracks_id with detection_id

threshold: float
if cost > threshold, it will not be considered

meethod: str global/ local

----------
returns
----------
row_idx: list of matched tracks (<= N) assigned tracklets' id

clo_idx: list of matched detections (<= M) assogned detections' id

unmatched_rows: list of unmatched ttracks, unassigned tracklets' id

unmatched_cols: list of unmatched detections. unassigned detections' id
'''
import numpy as np
def GreedyAssignment(cost, threshold = None, method = "global"):
    cost_c = np.atleast_2d(cost)
    size = cost_c.shape

    if threshold is None:
        threshold = 1.0

    row_idx = []
    col_idx = []

    if method == 'global':
        vector_in = list(range(size[0]))
        vector_out = list(range(size[1]))

        while min(len(vector_in),len(vector_out)) > 0:
            v = cost_c[np.ix_(vector_in, vector_out)]
            min_cost = np.min(v)

            if min_cost <= threshold:
                place = np.where(v == min_cost)
                row_idx.append(vector_in[place[0][0]])
                col_idx.append(vector_out[place[1][0]])
                del vector_in[place[0][0]]
                del vector_out[place[1][0]]
            else:
                break

    else: # local
        vector_in = []
        vector_out = list(range(size[1]))
        index = 0
        while min(size[0] - len(vector_in), len(vector_out)) > 0:
            if index >= size[0]:
                break
            place = np.argmin(cost_c[np.ix_([index], vector_out)])

            if cost_c[index, vector_out[place]] <= threshold:
                row_idx.append(index)
                col_idx.append(vector_out[place])
                del vector_out[place]
            else:
                vector_in.append(index)
            index += 1
        vector_in += list(range(index, size[0]))

    return np.array(row_idx), np.array(col_idx), np.array(vector_in), np.array(vector_out)