# We don't care about file batcher now...

from ..lib.signal_util import FileBatcher

from ..util.file_selector import FileSelector

fileSelector = FileSelector()
selectedFile = fileSelector.getSelectedFile(r"C:\Users\Gowtham\Documents\programs\HeartRate4\data\means")

batcher = FileBatcher(selectedFile, batchSize=2, bufSize=4)
for i in range(40):
    thisBatch = batcher.getNextBatch()
    print(thisBatch)
