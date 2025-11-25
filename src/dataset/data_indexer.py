from datetime import datetime
import h5py as h5


class DataIndexer:
    def __init__(self, input_dir, val_day, test_day, case_day=[]):
        self.dir = input_dir

        self.val_day = val_day
        self.test_day = test_day
        self.case_day = case_day

        self.train_index = []
        self.val_index = []
        self.test_index = []

        self._build_index()

    def _build_index(self):
        for file in sorted(self.dir.glob("*.h5")):
            with h5.File(file, "r") as f:
                times = [datetime.fromisoformat(t.decode("utf-8")) for t in f["time"]]

                for i, t in enumerate(times):
                    date_str = t.strftime("%Y%m%d")
                    day = t.day

                    entry = {"file": file, "index": i}

                    if date_str in self.case_day or day in self.test_day:
                        self.test_index.append(entry)
                    elif day in self.val_day:
                        self.val_index.append(entry)
                    else:
                        self.train_index.append(entry)
