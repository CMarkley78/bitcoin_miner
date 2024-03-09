import ctypes
import socket

print ("Client started.\n")
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

def get_work():
    print ("Getting work from server...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ('localhost', 50089)
    client_socket.connect(server_address)
    client_socket.sendall(bytes.fromhex("00"))
    header = client_socket.recv(76)
    id = client_socket.recv(4)
    client_socket.close()
    return header, id

def submit_nonce(nonce,id):
    print ("Submitting nonce...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 50089)
    client_socket.connect(server_address)
    client_socket.sendall(bytes.fromhex("01")+nonce+id)
    client_socket.close()
    print ("Submitted nonce.")

def shutdown_server():
    print ("Shutting down server...\n")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 50089)
    client_socket.connect(server_address)
    client_socket.sendall(bytes.fromhex("02"))
    client_socket.close()

#Loading in the dynamic library to handle the GPU
print ("Loading DLL...")
GPU_Handler = ctypes.CDLL(r"./build/GPU_Miner.dll")
print ("Loaded.\n")

if False:
    shutdown_server()
    exit()

tested = 0
nonces_found = 0
print ("Starting main loop...")
while True:
    tested += 1
    print (f"\nThis will be work task #{tested}... This machine has found {nonces_found} valid nonces.")
    header, id = get_work()
    result = test_header(header)
    if result[0]!=0:
        print ("Valid nonce was found!")
        nonces_found += 1
        submit_nonce(result[1],id)
    else:
        print ("No valid nonce was found.")
