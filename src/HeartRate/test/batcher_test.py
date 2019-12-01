from ..lib.signal_processor import Batcher

class DummyFrameProcessor:

    def __init__(self):
        self.x = [0.0, 0.0, 0.0]
    
    def getNextValues(self):
        self.x = [i+1 for i in self.x]
        # print(f"giving:{self.x}")
        return self.x

batcher = Batcher(DummyFrameProcessor(), batchSize=4, bufSize=4, debug=True)
for i in range(10):
    thisBatch = batcher.getNextBatch()
    print(thisBatch)

