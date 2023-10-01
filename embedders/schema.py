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
        self.iter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches
    
    def __next__(self) -> Tuple[int, List[slice]]:
        if self.iter < self.num_batches:
            iteration = (self.current_batch + self.iter, self.batch_list[self.current_batch + self.iter])
            self.iter += 1
            return iteration
        else:
            raise StopIteration
        
    def set_local_rank(self, rank: int, num_rank: int):
        '''
        rank start from 0
        num_ranks refer to `world_size`
        '''
        assert isinstance(rank, int)
        assert isinstance(num_rank, int)
        assert rank >= 0
        assert num_rank > rank

        # split data into chunk
        rank_size = int(self.num_batches/num_rank)
        start_position = rank_size*rank
        # add residue
        if (rank + 1) == num_rank:
            stop_position = self.num_batches
        else:
            stop_position = start_position + rank_size
        rank_num_batches = stop_position - start_position
        self.num_batches = rank_num_batches
        self.current_batch = start_position
