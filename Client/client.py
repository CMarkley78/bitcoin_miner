import ctypes

print ("Client started.")
def test_header(header):
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

    return output_array[0]==1, int.from_bytes(bytes(output_array[-4:]),byteorder="little",signed=False)

#Loading in the dynamic library to handle the GPU
print ("Loading DLL...")
GPU_Handler = ctypes.CDLL(r"./build/GPU_Miner.dll")

#Genesis header
sample_bytes = bytes.fromhex("0100000000000000000000000000000000000000000000000000000000000000000000003BA3EDFD7A7B12B27AC72C3E67768F617FC81BC3888A51323A9FB8AA4B1E5E4A29AB5F49FFFF001D")


print (test_header(sample_bytes))
