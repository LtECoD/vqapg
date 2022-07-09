import tempfile
from fairseq.data import IndexedRawTextDataset


def open_file(fn):
    try:
        with open(fn, "r") as f:
            lines = f.readlines()
        if len(lines) > 0:
            return [l.strip() for l in lines]
        else:
            print(f"{fn} is an empty file!")
            return None
    except FileNotFoundError:
        print(f'{fn} not exists!')
        return None


def build_indexedrawdataset_from_strlist(str_list, dictionary):
    fp = tempfile.NamedTemporaryFile(mode="w", buffering=1)
    fp.writelines([l + "\n" for l in str_list])

    raw_dataset = IndexedRawTextDataset(
        path=fp.name,
        dictionary=dictionary,
        append_eos=False,
    )

    fp.close()
    return raw_dataset
