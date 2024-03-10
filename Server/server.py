import socket
import binascii
import hashlib
from bitcoinrpc.authproxy import AuthServiceProxy
from random import randint, choice
from time import time


print ("Server started.")
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('127.0.0.1', 50089)
print('Starting up on {} port {}'.format(*server_address))
server_socket.bind(server_address)
server_socket.listen(10)

def create_work():
    return bytes.fromhex("01000000")+bytes.fromhex("0000000000000000000000000000000000000000000000000000000000000000")+bytes.fromhex("".join([choice([x for x in "0123456789ABCDEF"]) for _ in range(64)]))+bytes.fromhex("29AB5F495DBA0317")+bytes.fromhex("00000001")
    return bytes.fromhex("01000000")+bytes.fromhex("0000000000000000000000000000000000000000000000000000000000000000")+bytes.fromhex("".join([choice([x for x in "0123456789ABCDEF"]) for _ in range(64)]))+bytes.fromhex("29AB5F49FFFF001D")+bytes.fromhex("00000001")
    return bytes.fromhex("01000000")+bytes.fromhex("0000000000000000000000000000000000000000000000000000000000000000")+bytes.fromhex("3BA3EDFD7A7B12B27AC72C3E67768F617FC81BC3888A51323A9FB8AA4B1E5E4A")+bytes.fromhex("29AB5F49FFFF001D")+bytes.fromhex("00000001")

def difficulty_to_nbits(difficulty):
    difficulty = float(difficulty)
    hex_str = "00ffff"+("0"*26)
    return hex(int(float(int(hex_str,16))/difficulty))


def hashIt(firstTxHash, secondTxHash):
    unhex_reverse_first = binascii.unhexlify(firstTxHash)[::-1]
    unhex_reverse_second = binascii.unhexlify(secondTxHash)[::-1]

    concat_inputs = unhex_reverse_first+unhex_reverse_second
    first_hash_inputs = hashlib.sha256(concat_inputs).digest()
    final_hash_inputs = hashlib.sha256(first_hash_inputs).digest()
    return binascii.hexlify(final_hash_inputs[::-1])

def merkleCalculator(hashList):
    if len(hashList) == 1:
        return hashList[0]
    newHashList = []
    for i in range(0, len(hashList)-1, 2):
        newHashList.append(hashIt(hashList[i], hashList[i+1]))
    if len(hashList) % 2 == 1:
        newHashList.append(hashIt(hashList[-1], hashList[-1]))
    return merkleCalculator(newHashList)

def create_work_2():
    #Connect to Bitcoin Core
    rpc_connection = AuthServiceProxy("http://mark:rpcpasswd@127.0.0.1:8332")

    #Start header generation, init with version
    header = bytes.fromhex("04000000")

    #Add prevhash
    header = header+bytes.fromhex(rpc_connection.getbestblockhash())

    print (difficulty_to_nbits(rpc_connection.getdifficulty()))

solved = 0
while True:
    print (f"\nWaiting for client to send a message... So far, {solved} block(s) have been solved.")
    connection, client_address = server_socket.accept()
    data = connection.recv(1)
    if data[0]==0:
        print ("Client requested work, will be assigned.")
        connection.sendall(create_work())
    if data[0]==1:
        solved += 1
        print ("Client submitted a valid nonce!")
        nonce = connection.recv(4)
        work_id = connection.recv(4)
        print (nonce.hex(),work_id.hex())
    if data[0]==2:
        print ("Client sent shutdown signal... Goodbye!")
        connection.close()
        exit()
    connection.close()

create_work_2()
