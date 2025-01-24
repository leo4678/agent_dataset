import numpy as np
import math

# 将一个批次的样本合并成一个批次的张量
def collate_fn(examples):
    print(len(examples))
    for example in examples:
        print("00000")
        print(example["pixel_values"])
    print("----------------------------------")
    pixel_values_list = [np.array(example["pixel_values"]) for example in examples]
    pixel_values = np.stack(pixel_values_list).astype(np.float32)

    input_ids = [example["input_ids"] for example in examples]
    return {"pixel_values": pixel_values, "input_ids": input_ids}

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        print(len(self.dataset))
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield collate_fn(batch)