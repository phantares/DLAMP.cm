import h5py


def write_file(
    input_file,
    output_file,
    new_coords,
    predictions,
    targets,
    var_names,
):
    with h5py.File(input_file, "r") as f_in, h5py.File(output_file, "w") as f_out:

        for attr_name, attr_val in f_in.attrs.items():
            f_out.attrs[attr_name] = attr_val

        dims = []
        for dim in new_coords.keys():
            if dim in f_in:
                dset = f_out.create_dataset(dim, data=new_coords[dim])

                for a_name, a_val in f_in[dim].attrs.items():
                    dset.attrs[a_name] = a_val

                f_out[dim].make_scale(dim)
                dims.append(f_out[dim])

        pred_grp = f_out.create_group("predictions")
        targ_grp = f_out.create_group("targets")

        for i, name in enumerate(var_names):
            p_ds = pred_grp.create_dataset(
                name, data=predictions[:, i, ...], compression="gzip"
            )
            attach_dim(p_ds, *dims)

            t_ds = targ_grp.create_dataset(
                name, data=targets[:, i, ...], compression="gzip"
            )
            attach_dim(t_ds, *dims)

            if name in f_in:
                for a_n, a_v in f_in[name].attrs.items():
                    p_ds.attrs[a_n] = a_v
                    t_ds.attrs[a_n] = a_v

    print(f"File saved : {output_file}")


def attach_dim(data, *dims):
    for d, dim in enumerate(dims):
        data.dims[d].attach_scale(dim)


def print_h5_structure(file):
    with h5py.File(file, "r") as f:
        print(f"File: {file}")
        f.visititems(print_variables)


def print_variables(name, obj):
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📊 Dataset: {name}")
        print(f"{indent}    Shape: {obj.shape}")
        print(f"{indent}    Dtype: {obj.dtype}")
        if obj.attrs:
            print(f"{indent}    Attrs: {dict(obj.attrs)}")
