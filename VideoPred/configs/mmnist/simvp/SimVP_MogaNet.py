method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'moga'
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
# training
lr = 1e-3/(2**0.5)
batch_size = 4
drop_path = 0
sched = 'onecycle'