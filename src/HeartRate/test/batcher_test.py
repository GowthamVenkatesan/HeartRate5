from ..lib.signal_processor import Batcher, DummyFrameProcessor

batcher = Batcher(DummyFrameProcessor(), batchSizeT=4, bufSizeT=4, debug=True)
for i in range(10):
    thisBatch = batcher.getNextBatch()
    print(thisBatch)

