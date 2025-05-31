import torch
from torch.utils.data import (
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)
from collections import defaultdict

class MixedBatchSampler(BatchSampler):
    """Sample one batch from a selected dataset with given probability.
    Compatible with datasets at different resolution
    """

    def __init__(
        self, src_dataset_ls,
        accumulation_steps,
        batch_size, drop_last, 
        shuffle, 
        iterative_sampling=True,
        prob=None, generator=None
    ):
        self.base_sampler = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        self.accumulation_steps = accumulation_steps
        self.src_dataset_ls = src_dataset_ls
        self.n_dataset = len(self.src_dataset_ls)
        self.iterative_sampling = iterative_sampling
        print(f"Iterative sampling: {iterative_sampling}")
        
        # Dataset length
        self.dataset_length = [len(ds) for ds in self.src_dataset_ls]
        self.cum_dataset_length = [
            sum(self.dataset_length[:i]) for i in range(self.n_dataset)
        ]  # cumulative dataset length


        # BatchSamplers for each source dataset
        if self.shuffle:
            self.src_batch_samplers = [
                BatchSampler(
                    sampler=RandomSampler(
                        ds, replacement=False, generator=self.generator
                    ),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                )
                for ds in self.src_dataset_ls
            ]
        else:
            self.src_batch_samplers = [
                BatchSampler(
                    sampler=SequentialSampler(ds),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                )
                for ds in self.src_dataset_ls
            ]
        self.raw_batches = [
            list(bs) for bs in self.src_batch_samplers
        ]  # index in original dataset
        self.n_batches = [len(b) for b in self.raw_batches]
        self.n_total_batch = sum(self.n_batches)

        # sampling probability
        if prob is None:
            # if not given, decide by dataset length
            self.prob = torch.tensor(self.n_batches) / self.n_total_batch
        else:
            self.prob = torch.as_tensor(prob)


        self.tasks = defaultdict(list)
        for i, ds in enumerate(self.src_dataset_ls):
            self.tasks[ds.output_type].append({
                "idx_ds": i,
                "prob": self.prob[i],
                "name": ds.disp_name
            })
        self.tasks_keys = list(self.tasks.keys())

    def __iter__(self):
        """_summary_

        Yields:
            list(int): a batch of indics, corresponding to ConcatDataset of src_dataset_ls
        """
        for batch_idx in range(self.n_total_batch):
            effective_batch_idx = batch_idx // self.accumulation_steps
            n_tasks = len(self.tasks_keys)
            if self.iterative_sampling:
                task_idx = effective_batch_idx % n_tasks
            else:
                task_idx = torch.randint(n_tasks, (1,), generator=self.generator).item()
            # print("Task: ", self.tasks_keys[task_idx])
            task_info = self.tasks[self.tasks_keys[task_idx]]
            probs = [ds["prob"] for ds in task_info]
            probs = torch.tensor(probs) / sum(probs)
            # print([ds["name"] for ds in task_info])
            # print("Probs: ", probs)
            idx_ds_of_this_task = torch.multinomial(probs, 1, replacement=True, generator=self.generator).item()
            # print("Idx ds of this task: ", idx_ds_of_this_task)
            idx_ds = task_info[idx_ds_of_this_task]["idx_ds"]
            # if batch list is empty, generate new list
            if 0 == len(self.raw_batches[idx_ds]):
                self.raw_batches[idx_ds] = list(self.src_batch_samplers[idx_ds])
            # get a batch from list
            batch_raw = self.raw_batches[idx_ds].pop()
            # shift by cumulative dataset length
            shift = self.cum_dataset_length[idx_ds]
            batch = [n + shift for n in batch_raw]

            yield batch

    def __len__(self):
        return self.n_total_batch


# Unit test
if "__main__" == __name__:
    from torch.utils.data import ConcatDataset, DataLoader, Dataset

    class SimpleDataset(Dataset):
        def __init__(self, start, len) -> None:
            super().__init__()
            self.start = start
            self.len = len

        def __len__(self):
            return self.len

        def __getitem__(self, index):
            return self.start + index

    dataset_1 = SimpleDataset(0, 10)
    dataset_2 = SimpleDataset(200, 20)
    dataset_3 = SimpleDataset(1000, 50)

    concat_dataset = ConcatDataset(
        [dataset_1, dataset_2, dataset_3]
    )  # will directly concatenate

    mixed_sampler = MixedBatchSampler(
        src_dataset_ls=[dataset_1, dataset_2, dataset_3],
        batch_size=4,
        drop_last=True,
        shuffle=False,
        prob=[0.6, 0.3, 0.1],
        accumulation_steps=2,
        generator=torch.Generator().manual_seed(0),
    )

    loader = DataLoader(concat_dataset, batch_sampler=mixed_sampler)

    for d in loader:
        print(d)
