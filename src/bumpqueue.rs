use crossbeam::queue::SegQueue;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const BLOCK_SIZE: usize = 1024;
const BLOCK_ALLOC_TRIGGER: usize = BLOCK_SIZE - 1;

// A block of data for the Queue and Pool below.
struct BumpBlock<T> {
    data: Box<[UnsafeCell<MaybeUninit<T>>; BLOCK_SIZE]>,
}

impl<T> BumpBlock<T> {
    pub fn new() -> Self {
        Self {
            data: Box::new(std::array::from_fn(|_| {
                UnsafeCell::new(MaybeUninit::uninit())
            })),
        }
    }

    pub fn write(&self, index: usize, value: T) {
        let cell = &self.data[index];
        unsafe {
            std::ptr::write(cell.get(), MaybeUninit::new(value));
        }
    }

    pub fn read(&self, index: usize) -> T {
        let cell = &self.data[index];
        unsafe { std::ptr::read(cell.get() as *const T) }
    }
}

// The pool for the Queue implementation below.
pub struct BumpPool<T> {
    freelist: SegQueue<Arc<BumpBlock<T>>>,
}

unsafe impl<T: Send> Send for BumpPool<T> {}
unsafe impl<T: Send> Sync for BumpPool<T> {}

impl<T> BumpPool<T> {
    pub fn new() -> Self {
        Self {
            freelist: SegQueue::new(),
        }
    }

    fn alloc_block(&self) -> Arc<BumpBlock<T>> {
        if let Some(block) = self.freelist.pop() {
            block
        } else {
            Arc::new(BumpBlock::new())
        }
    }

    fn return_block(&self, block: Arc<BumpBlock<T>>) {
        self.freelist.push(block);
    }
}

// A Multiple-Producer Single-Consumer queue which uses a pool of blocks to store the data.
// This data structure is optimized for the following usage pattern ONLY
//   1. Concurrent push of an unknown number of elements
//   2. Single-threaded pop consuming the queue once
//   3. Reset
// This queue is used instead of the excellent crossbeam::SegQueue for two reasons:
//   It's slightly faster and simpler because of the limited use case
//   It load-balances backing memory (blocks) between queues, which saves memory when the workload changes between frames
pub struct BumpQueue<T> {
    pool: Arc<BumpPool<T>>,
    blocks: boxcar::Vec<Arc<BumpBlock<T>>>,
    count: AtomicUsize,
}

impl<T> BumpQueue<T> {
    pub fn new(pool: Arc<BumpPool<T>>) -> Self {
        Self {
            pool,
            blocks: boxcar::Vec::new(),
            count: AtomicUsize::new(0),
        }
    }

    pub fn push(&self, value: T) {
        let backoff = crossbeam_utils::Backoff::new();
        let idx = self.count.fetch_add(1, Ordering::Relaxed);
        let block_idx = idx / BLOCK_SIZE;
        let local_idx = idx % BLOCK_SIZE;

        if idx == 0 {
            // First push, allocate a block
            self.blocks.push(self.pool.alloc_block());
        } else if local_idx == BLOCK_ALLOC_TRIGGER {
            // Allocate a new block expecting the next push
            self.blocks.push(self.pool.alloc_block());
        }

        // Spin until we have enough blocks
        while self.blocks.count() <= block_idx {
            // Waiting for the previous push to add a block
            backoff.snooze();
        }

        // Store the value
        let block = &self.blocks[block_idx];
        block.write(local_idx, value);
    }

    pub fn get(&self, index: usize) -> T {
        debug_assert!(index < self.len());
        let block_idx = index / BLOCK_SIZE;
        let local_idx = index % BLOCK_SIZE;
        let block = &self.blocks[block_idx];
        block.read(local_idx)
    }

    // Return all the blocks to the pool and reset the queue
    pub fn reset(&mut self) {
        for (_, block) in &self.blocks {
            self.pool.return_block(block.clone());
        }
        self.blocks.clear();
        self.count.store(0, Ordering::Relaxed);
    }

    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    // Sorts the queue in place using the provided comparison function.
    pub fn sort_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&T, &T) -> std::cmp::Ordering,
    {
        let count = self.len();
        if count <= 1 {
            return; // Nothing to sort
        }

        // Perform in-place quicksort directly on the blocks
        self.quicksort_range(0, count - 1, &mut compare);
    }

    // Helper method to perform in-place quicksort on a range of indices
    fn quicksort_range<F>(&mut self, low: usize, high: usize, compare: &mut F)
    where
        F: FnMut(&T, &T) -> std::cmp::Ordering,
    {
        if low < high {
            let pivot_index = self.partition(low, high, compare);
            if pivot_index > 0 {
                self.quicksort_range(low, pivot_index - 1, compare);
            }
            if pivot_index + 1 < high {
                self.quicksort_range(pivot_index + 1, high, compare);
            }
        }
    }

    // Helper method to partition the range for quicksort
    fn partition<F>(&mut self, low: usize, high: usize, compare: &mut F) -> usize
    where
        F: FnMut(&T, &T) -> std::cmp::Ordering,
    {
        // Choose the rightmost element as pivot
        let pivot = self.get_value(high);
        let mut i = low;

        for j in low..high {
            if compare(&self.get_value(j), &pivot) == std::cmp::Ordering::Less {
                self.swap_values(i, j);
                i += 1;
            }
        }

        self.swap_values(i, high);
        i
    }

    // Helper method to get a value at a specific index
    fn get_value(&self, index: usize) -> T {
        let block_idx = index / BLOCK_SIZE;
        let local_idx = index % BLOCK_SIZE;
        let block = &self.blocks[block_idx];
        block.read(local_idx)
    }

    // Helper method to swap two values at specific indices
    fn swap_values(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }

        let temp = self.get_value(i);
        let value_j = self.get_value(j);

        // Write value_j to position i
        let block_idx_i = i / BLOCK_SIZE;
        let local_idx_i = i % BLOCK_SIZE;
        let block_i = &self.blocks[block_idx_i];
        block_i.write(local_idx_i, value_j);

        // Write temp to position j
        let block_idx_j = j / BLOCK_SIZE;
        let local_idx_j = j % BLOCK_SIZE;
        let block_j = &self.blocks[block_idx_j];
        block_j.write(local_idx_j, temp);
    }
}

// Don't worry, we've got this, probably
unsafe impl<T: Send> Send for BumpQueue<T> {}
unsafe impl<T: Send> Sync for BumpQueue<T> {}
