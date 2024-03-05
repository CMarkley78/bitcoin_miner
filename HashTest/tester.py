import ctypes
from time import time

def test_header(header):
    print ("Working...")
    header_template_array = (ctypes.c_ubyte*76)(*header)
    header_template_pointer = ctypes.cast(header_template_array,ctypes.POINTER(ctypes.c_ubyte))

    #Create an array and pointer for the output
    output_array = (ctypes.c_ubyte*5)()
    output_pointer = ctypes.cast(output_array,ctypes.POINTER(ctypes.c_ubyte))

    #Get the function ready
    search = GPU_Handler.search
    search.argtypes = [ctypes.POINTER(ctypes.c_ubyte),ctypes.POINTER(ctypes.c_ubyte)]

    #Call the function
    search(header_template_pointer,output_pointer)

    #Return the results
    return output_array[0]==1, bytes(output_array[-4:])

print ("Started hashrate tester.\nLoading DLL...")
GPU_Handler = ctypes.CDLL(r"./build/GPU_Miner.dll")
print ("Loaded.\n")

start = time()
result = test_header(bytes.fromhex("0100000000000000000000000000000000000000000000000000000000000000000000003BA3EDFD7A7B12B27AC72C3E67768F617FC81BC3888A51323A9FB8AA4B1E5E4A29AB5F49FFFF001D"))
end = time()

print (f"\nSpan search executed in {end-start}s.\nIt returned                  {(result[0],int.from_bytes(result[1], "little"))}.\nThe expected return value is (True, 2083236893).\nThis hasher is {'correct' if ((result[0],int.from_bytes(result[1], "little"))==(True, 2083236893)) else 'incorrect'}.\nThe hashrate of this hasher is {((2**32)/(end-start))/10**6}MH/s.")
