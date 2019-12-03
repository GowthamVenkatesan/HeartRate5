from ..lib.batcher import Batcher
from ..lib.frame_processor import DummyFrameProcessor

batcher = Batcher(DummyFrameProcessor(), batchSizeT=4, bufSizeT=4, debug=True)
for i in range(10):
    thisBatch = batcher.getNextBatch()
    print(thisBatch)

