import numpy as np
import torch
from torch.utils.data import TensorDataset


# # -------------------------------
# # One-hot encoding
# # -------------------------------
# def one_hot_encode(seq):
#     """
#     Convert DNA sequence to one-hot matrix
#     Shape: (len(seq), 4) -> A,C,G,T
#     """
#     mapping = {"A":0,"C":1,"G":2,"T":3}
#     arr = np.zeros((len(seq),4),dtype=np.int8)
#     for i,base in enumerate(seq.upper()):
#         if base in mapping:
#             arr[i,mapping[base]] = 1
#     return arr


# # -------------------------------
# # Initial sequence
# # -------------------------------
# def initial_sequence_from_df(
#         df,
#         sequence_col,
#         n_seq,
#         mode="sample",        # "sample" or "generate"
#         seq_len=None
#     ):
#     """
#     Generate sequence set from dataframe.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#     sequence_col : str
#         Column name containing DNA sequences
#     n_seq : int
#         Number of sequences to return
#     mode : str
#         "sample"  -> randomly sample real sequences
#         "generate"-> generate new sequences by background frequency
#     seq_len : int
#         Required length of sequences

#     Returns
#     -------
#     random_regions : np.array
#         Shape (n_seq, seq_len, 4)
#     """

#     # Extract sequences
#     seqs = df[sequence_col].astype(str).str.upper().tolist()

#     # Length checking
#     lengths = [len(s) for s in seqs]
#     if seq_len is None:
#         seq_len = lengths[0]

#     if not all(l==seq_len for l in lengths):
#         raise ValueError("Input sequences have inconsistent length")

#     if mode=="sample":

#         if n_seq>len(seqs):
#             raise ValueError("n_seq exceeds available sequences")

#         # Random sampling
#         chosen = np.random.choice(seqs,n_seq,replace=False)
#         random_regions = [one_hot_encode(s) for s in chosen]


#     elif mode=="generate":

#         mapping={"A":0,"C":1,"G":2,"T":3}

#         # Calculate position-specific distribution
#         ACGT_dist = np.zeros((seq_len,4))

#         for s in seqs:
#             for i,b in enumerate(s):
#                 if b in mapping:
#                     ACGT_dist[i,mapping[b]] += 1

#         ACGT_dist = ACGT_dist / ACGT_dist.sum(axis=1,keepdims=True)

#         # Generate sequences
#         random_regions=[]
#         for _ in range(n_seq):
#             seq=[]
#             for i in range(seq_len):
#                 seq.append(np.random.choice(
#                     ["A","C","G","T"],
#                     p=ACGT_dist[i]
#                 ))
#             random_regions.append(one_hot_encode("".join(seq)))

#     else:
#         raise ValueError("mode must be 'sample' or 'generate'")

#     random_regions = np.array(random_regions)
#     return random_regions

# """
# X = initial_sequence_from_df(
#         df,
#         sequence_col="sequence",
#         n_seq=1000,
#         mode="sample"
# )

# X = initial_sequence_from_df(
#         df,
#         sequence_col="sequence",
#         n_seq=1000,
#         mode="generate"
# )
# """


# -------------------------------
# In-silico evolution
# -------------------------------
def create_saturation_mutagenesis_x(onehot):
    mutagenesis_X = {"X":[],"ids":[]}
    onehot = onehot.squeeze()
    for mutloc,nt in enumerate(onehot):
        new_X = np.copy(onehot)
        if list(nt) == [1, 0, 0, 0]:
            new_X[mutloc,:] = np.array([0, 1, 0, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 0, 1, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 0, 0, 1], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            mutagenesis_X["ids"].append(str(mutloc)+"_C")
            mutagenesis_X["ids"].append(str(mutloc)+"_G")
            mutagenesis_X["ids"].append(str(mutloc)+"_T")
        if list(nt) == [0, 1, 0, 0]:
            new_X[mutloc,:] = np.array([1, 0, 0, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 0, 1, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 0, 0, 1], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            mutagenesis_X["ids"].append(str(mutloc)+"_A")
            mutagenesis_X["ids"].append(str(mutloc)+"_G")
            mutagenesis_X["ids"].append(str(mutloc)+"_T")
        if list(nt) == [0, 0, 1, 0]:
            new_X[mutloc,:] = np.array([1, 0, 0, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 1, 0, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 0, 0, 1], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            mutagenesis_X["ids"].append(str(mutloc)+"_A")
            mutagenesis_X["ids"].append(str(mutloc)+"_C")
            mutagenesis_X["ids"].append(str(mutloc)+"_T")
        if list(nt) == [0, 0, 0, 1]:
            new_X[mutloc,:] = np.array([1, 0, 0, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 1, 0, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            new_X[mutloc,:] = np.array([0, 0, 1, 0], dtype='int8')
            mutagenesis_X["X"].append(np.copy(new_X))
            mutagenesis_X["ids"].append(str(mutloc)+"_A")
            mutagenesis_X["ids"].append(str(mutloc)+"_C")
            mutagenesis_X["ids"].append(str(mutloc)+"_G")
            
    mutagenesis_X["X"] = np.array(mutagenesis_X["X"])
    return mutagenesis_X 


def insilico_evolution(regions, model, class_no, n_mutation, device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # guide_model = model[class_no]

    #from scipy.stats import zscore
    nuc_to_onehot = {"A":[1, 0, 0, 0],"C":[0, 1, 0, 0],"G":[0, 0, 1, 0],"T":[0, 0, 0, 1]}
    mutation_pred = []
    mutation_loc = []
    print("Sequence index:",end=" ")
    for id_ in range(len(regions)):
        start_x = np.copy(regions[id_:id_+1])
        pred = []
        mut = []
        for i in range(n_mutation):
            mutagenesis_X = create_saturation_mutagenesis_x(start_x)

            x = torch.from_numpy(mutagenesis_X["X"]).float().to(device)
            x = x.permute(0, 2, 1).contiguous()
            # dataset = TensorDataset(x)
            with torch.no_grad():
                # prediction_mutagenesis_X = guide_model.predict_on_dataset(dataset)
                prediction_mutagenesis_X = model(x)
            prediction_mutagenesis_X = prediction_mutagenesis_X.detach().cpu().numpy()

            x_ori = torch.from_numpy(start_x).float().to(device)
            x_ori = x_ori.permute(0, 2, 1).contiguous()
            # dataset_ori = TensorDataset(x_ori)
            with torch.no_grad():
                # original_prediction = guide_model.predict_on_dataset(dataset_ori)
                original_prediction = model(x_ori)
            original_prediction = original_prediction.detach().cpu().numpy()

            ## To use max z-score
            # next_one = mutagenesis_X["ids"][np.argmax(zscore(prediction_mutagenesis_X-original_prediction,axis=1)[:,class_no-1])]
            ## To use max score
            next_one = mutagenesis_X["ids"][np.argmax(prediction_mutagenesis_X[:,class_no]-original_prediction[:,class_no])]
            pred.append(original_prediction)
            mut.append(next_one)
            start_x[0][int(next_one.split("_")[0]),:] = np.array(nuc_to_onehot[next_one.split("_")[1]], dtype='int8')
         
        x_ori = torch.from_numpy(start_x).float().to(device)
        x_ori = x_ori.permute(0, 2, 1).contiguous()
        # dataset_ori = TensorDataset(x_ori)
        with torch.no_grad():
            # original_prediction = guide_model.predict_on_dataset(dataset_ori)
            original_prediction = model(x_ori)
        original_prediction = original_prediction.detach().cpu().numpy()

        pred.append(original_prediction)
        mutation_pred.append(pred)
        mutation_loc.append(mut)
        print(id_,end=",")
    mutation_pred = np.array(mutation_pred).squeeze()
    mutation_loc = np.array(mutation_loc)
    return mutation_pred, mutation_loc







def one_hot_encode_along_row_axis(sequence):
    to_return = np.zeros((1, len(sequence), 4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return[0], sequence=sequence, one_hot_axis=1)
    return to_return


def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis == 0 or one_hot_axis == 1
    if one_hot_axis == 0:
        assert zeros_array.shape[1] == len(sequence)
    elif one_hot_axis == 1:
        assert zeros_array.shape[0] == len(sequence)
    for (i, char) in enumerate(sequence):
        if char == "A" or char == "a":
            char_idx = 0
        elif char == "C" or char == "c":
            char_idx = 1
        elif char == "G" or char == "g":
            char_idx = 2
        elif char == "T" or char == "t":
            char_idx = 3
        elif char == "N" or char == "n":
            continue
        else:
            raise RuntimeError("Unsupported character: " + str(char))
        if one_hot_axis == 0:
            zeros_array[char_idx, i] = 1
        elif one_hot_axis == 1:
            zeros_array[i, char_idx] = 1

########################################
## motif implantation
########################################
def add_pattern_to_best_location(pattern, regions, model, class_no, device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    pattern_added_regions =  np.zeros(regions.shape,dtype="int")
    pattern_locations = np.zeros(regions.shape[0],dtype="int")
    print("Sequence index:",end=" ")
    for r, region in enumerate(regions):
        tmp_array = np.zeros((regions.shape[1]-pattern.shape[1]+1,regions.shape[1],regions.shape[2]))
        for nt in range(tmp_array.shape[0]):
            tmp_array[nt] = np.copy(region)
            tmp_array[nt,nt:nt+pattern.shape[1],:] = pattern[0]
        # prediction = model.predict(tmp_array)[:,class_no]
        tmp_array = torch.from_numpy(tmp_array).float().to(device)
        with torch.no_grad():
            prediction = model(tmp_array.permute(0, 2, 1).contiguous())
        prediction = prediction.detach().cpu().numpy()
        pattern_locations[r] = np.argmax(prediction)    
        pattern_added_regions[r] = tmp_array[pattern_locations[r]].cpu().numpy()
        print(r,end=",")
    print("")
    return {"regions":pattern_added_regions, "locations":pattern_locations}


import matplotlib


def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.2, 0.0],
        ]),
        np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.8, 0.0],
        ]),
        np.array([
            [0.225, 0.45],
            [0.775, 0.45],
            [0.85, 0.3],
            [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.825, base + 0.085 * height], width=0.174, height=0.415 * height,
                                     facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.625, base + 0.35 * height], width=0.374, height=0.15 * height,
                                     facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.4, base],
                                              width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.8 * height],
                                              width=1.0, height=0.2 * height, facecolor=color, edgecolor=color,
                                              fill=True))


default_colors = {0: 'green', 1: 'blue', 2: 'orange', 3: 'red'}
default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t}


def plot_weights_given_ax(ax, array,
                          height_padding_factor,
                          length_padding,
                          subticks_frequency,
                          highlight,
                          colors=default_colors,
                          plot_funcs=default_plot_funcs):
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if array.shape[0] == 4 and array.shape[1] != 4:
        array = array.transpose(1, 0)
    assert array.shape[1] == 4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if letter[1] > 0:
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos, min_depth],
                                             width=end_pos - start_pos,
                                             height=max_height - min_depth,
                                             edgecolor=color, fill=False))

    ax.set_xlim(-length_padding, array.shape[0] + length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, subticks_frequency))
    height_padding = max(abs(min_neg_height) * (height_padding_factor),
                         abs(max_pos_height) * (height_padding_factor))
    ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)
    return ax



def plot_weights(array, fig, n, n1, n2, title='', ylab='',
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=20,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={}):
    ax = fig.add_subplot(n, n1, n2)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    y = plot_weights_given_ax(ax=ax, array=array,
                              height_padding_factor=height_padding_factor,
                              length_padding=length_padding,
                              subticks_frequency=subticks_frequency,
                              colors=colors,
                              plot_funcs=plot_funcs,
                              highlight=highlight)
    return fig, ax