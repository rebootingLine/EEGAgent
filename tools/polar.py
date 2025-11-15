bipolar_pairs = [
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'),
    ('CZ', 'C4'), ('C4', 'T4'), ('T4', 'A2'), 
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
]

index2name = {i: f"{pair[0]}-{pair[1]}" for i, pair in enumerate(bipolar_pairs)}
index2name_ = {i: f"{pair[1]}-{pair[0]}" for i, pair in enumerate(bipolar_pairs)}
name2index = {**{v: k for k, v in index2name.items()},
              **{v: k for k, v in index2name_.items()}}


single_polar = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'F7', 'T3', 'T5', 'Fz', 'Fp2',
    'F4', 'C4', 'P4', 'O2', 'F8',
    'T4', 'T6', 'Cz', 'Pz'
]
single_index2name = {i: f"{name}" for i, name in enumerate(single_polar)}
single_name2index = {v: k for k, v in index2name.items()}


sleep_polar = ['Fpz-Cz', 'Pz-Oz']
sleep_index2name = {i: f"{name}" for i, name in enumerate(sleep_polar)}
sleep_name2index = {v: k for k, v in index2name.items()}