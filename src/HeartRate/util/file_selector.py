import os

class FileSelector:

    def __init__(self):
        pass
    
    def getSelectedFile(self, dirPath):
        files = os.listdir(dirPath)
        print("=" * 60)
        print(f"files:")
        print("-" * 60)
        for idx, f in enumerate(files, 1):
            print(f"{idx}. {f}")
        print("-" * 60)
        selectedFile = int(input("file: "))
        fullFilePath = os.path.join(dirPath, files[selectedFile-1])
        print("=" * 60)
        return fullFilePath