from typing import List, Iterable, Tuple

class BatchIterator:
    '''
    iterator with slices
    '''
    def __init__(self, batch_list: List[slice], start_batch: int) -> Iterable:
        assert isinstance(batch_list, list)
        assert isinstance(start_batch, int)
        self.batch_list = batch_list
        self.total_batches = len(batch_list)
        self.current_batch = start_batch
        batch_list_to_iterate = self.batch_list[self.current_batch:]
        self.num_batches = len(batch_list_to_iterate)

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches
    
    def __next__(self) -> Tuple[int, List[slice]]:
        if self.current_batch < self.total_batches:
            iteration = (self.current_batch, self.batch_list[self.current_batch])
            self.current_batch += 1
            return iteration
        else:
            raise StopIteration