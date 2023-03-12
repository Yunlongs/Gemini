import pickle


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


version = ["openssl-101a", "openssl-101f"]
arch = ["arm", "x86", "mips"]
compiler = ["clang", "gcc"]
optimizer = ["O0", "O1", "O2", "O3"]
dir_name = "data/extracted-acfg/"

for a in arch:
    num = 0
    for v in version:
        for c in compiler:
            for o in optimizer:
                filename = "_".join([v, a, c, o, "openssl"])
                filename += ".cfg"
                filename = dir_name + filename
                with open(filename, "r") as f:
                    picklefile = pickle.load(StrToBytes(f))
                num += len(picklefile.raw_graph_list)
    print(a, ":", num)
