use crossbeam::queue::SegQueue;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const BLOCK_SIZE: usize = 128;
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
            (*cell.get()).write(value);
        }
    }

    pub fn read(&self, index: usize) -> T {
        let cell = &self.data[index];
        unsafe { (*cell.get()).assume_init_read() }
    }
}

// The pool for the Queue implementation below.
pub struct BumpPool<T> {
    blocks: boxcar::Vec<BumpBlock<T>>,
    freelist: SegQueue<usize>,
}

unsafe impl<T: Send> Send for BumpPool<T> {}
unsafe impl<T: Send> Sync for BumpPool<T> {}

impl<T> BumpPool<T> {
    pub fn new() -> Self {
        Self {
            blocks: boxcar::Vec::new(),
            freelist: SegQueue::new(),
        }
    }

    fn alloc_block(&self) -> usize {
        if let Some(index) = self.freelist.pop() {
            index
        } else {
            self.blocks.push(BumpBlock::new())
        }
    }

    fn return_block(&self, index: usize) {
        self.freelist.push(index);
    }

    fn get_block(&self, index: usize) -> &BumpBlock<T> {
        &self.blocks[index]
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
    blocks: boxcar::Vec<usize>,
    count: AtomicUsize,
    head: usize,
}

impl<T> BumpQueue<T> {
    pub fn new(pool: Arc<BumpPool<T>>) -> Self {
        Self {
            pool,
            blocks: boxcar::Vec::new(),
            count: AtomicUsize::new(0),
            head: 0,
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
        let block = self.pool.get_block(self.blocks[block_idx]);
        block.write(local_idx, value);
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.head >= self.count.load(Ordering::Relaxed) {
            return None;
        }

        let idx = self.head;
        let block_idx = idx / BLOCK_SIZE;
        let local_idx = idx % BLOCK_SIZE;

        let block = self.pool.get_block(self.blocks[block_idx]);
        let value = block.read(local_idx);

        self.head = idx + 1;

        Some(value)
    }

    // Return all the blocks to the pool and reset the queue
    pub fn reset(&mut self) {
        for block in self.blocks.iter() {
            self.pool.return_block(*block.1);
        }
        self.blocks.clear();
        self.count.store(0, Ordering::Relaxed);
        self.head = 0;
    }
}

// Don't worry, we've got this, probably
unsafe impl<T: Send> Send for BumpQueue<T> {}
unsafe impl<T: Send> Sync for BumpQueue<T> {}
