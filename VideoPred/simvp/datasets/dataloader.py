

def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, **kwargs):
    pre_seq_length = kwargs.get('pre_seq_length', 10)
    aft_seq_length = kwargs.get('aft_seq_length', 10)
    if 'shapes' in dataname:
        from .dataloader_shapes import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
