import torch
from torch.nn.utils.rnn import pad_sequence
import math
from itertools import combinations

## Example data
## Video 1 with 10 frames, each frame represented by a 512-dimensional vector
#video1 = torch.randn(10, 3)
## Video 2 with 8 frames, each frame represented by a 512-dimensional vector
#video2 = torch.randn(8, 3)
#
## Padding sequences to the same length
#padded_video1 = pad_sequence(
#    [video1, video2], batch_first=True, padding_value=0.0)
#padded_video2 = pad_sequence(
#    [video2, video1], batch_first=True, padding_value=0.0)
#
## Mask for ignoring padded elements
#src_mask = torch.ones_like(padded_video1)
#src_mask[padded_video1 == 0] = 0
#print(src_mask)

#query_frame = torch.tensor([[[0.00, 0.01], [0.10, 0.11], [0.20, 0.21]], 
#                            [[1.00, 1.01], [1.10, 1.11], [1.20, 1.21]], 
#                            [[2.00, 2.01], [2.10, 2.11], [2.20, 2.21]], 
#                            [[3.00, 3.01], [3.10, 3.11], [3.20, 3.21]]])
#perm_query_frame = query_frame.permute(0,2,1)
#res_quer_frame = query_frame.reshape(4, -1, 6)
#cat_query_frame = torch.cat((query_frame, query_frame), 0)
#cat_query_frame = torch.cat((query_frame, query_frame), 1)
#cat_query_frame = torch.cat((query_frame, query_frame), 2)
#
#support_frame = query_frame
#
#class_scores = torch.matmul(support_frame, query_frame.transpose(1,0))
#class_scores = torch.softmax(class_scores, dim = 0)
#diff = class_scores - query_frame
#norm_sq = torch.norm(diff, dim=[-2, -1])**2
#distance = torch.div(norm_sq, 3)
#print(distance)

def create_mask(support_set_length, query_set_length, attention_map, tuples):
  for id, t in enumerate(tuples):
    if t[0] > support_set_length or t[1] > support_set_length:
      attention_map[:, id] = float('-inf')
    if t[0] > query_set_length or t[1] > query_set_length:
      attention_map[id, :] = float('-inf')

def create_mask_TRX(support_set_lengths, query_set_lengths, attention_maps, tuples):
  for id_q, q_length in enumerate(query_set_lengths):
    for id_s, s_length in enumerate(support_set_lengths):
      create_mask(s_length, q_length, attention_maps[id_q][id_s], tuples)


def delete_tuples(l, n, temporal_set_size):
    frame_idxs = [i for i in range(1, l+1)]
    frame_combinations = combinations(frame_idxs, temporal_set_size)
    cardinality_combs = [comb for comb in frame_combinations]
    id_to_delete_list = [id for id, t in enumerate(cardinality_combs) if any (x > n for x in t)]
    return id_to_delete_list

#rand_tensor = torch.randn(7, 256, 3)
#rand_tensor = torch.nn.functional.pad(rand_tensor, (0, 0, 0, 0, 0, 3), "constant", 0)
#
#seq_len = 16
#frame_idxs = [i for i in range(seq_len)]
#frame_combinations = combinations(frame_idxs, 2)
#tuples = [torch.tensor(comb) for comb in frame_combinations]
#tuples_len = len(tuples)
#attention_map1 = torch.randn(tuples_len, tuples_len)
#attention_map2 = torch.randn(tuples_len, tuples_len)
#attention_map3 = torch.randn(tuples_len, tuples_len)
#attention_map4 = torch.randn(tuples_len, tuples_len)
#support_set_lengths = [8,15]
#query_set_lengths = [14,10]
#attention_maps = torch.cat([attention_map1, attention_map2, attention_map3, attention_map4])
#attention_maps = attention_maps.reshape(2, 2, tuples_len, -1)
#create_mask_TRX(support_set_lengths, query_set_lengths, attention_maps, tuples)
#print(attention_maps.shape)

#seq_len = 100
#test = [delete_tuples(seq_len, i, 2) for i in range(0, seq_len+1)]
#print(f"96: {test[0]}")
#print(f"97: {test[1]}")
#print(f"98: {test[-1]}")
#
#test = delete_tuples(8, 5, 2)
#print(test)

class_softmax = torch.nn.Softmax(dim=0)
class_scores = torch.randn(5, 66, 10)
target_n_frames = [10,12,8,6,5]
tuples_mask = [torch.tensor(delete_tuples(12, n, 3)).cuda() for n in range(0, 13)]

#for i_q, query in enumerate(class_scores):
#    n_frames = target_n_frames[i_q]
#    mask = tuples_mask[n_frames-2]
#    for i_t, query in enumerate(query):
#        if i_t in mask:
#            class_scores[i_q][i_t] = 0
#        else:
#            query = class_softmax(query)
#            class_scores[i_q][i_t] = query

print("Test Ende")