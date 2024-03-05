import socket
import binascii
import hashlib
from bitcoinrpc.authproxy import AuthServiceProxy
from bitcoinlib.transactions import Transaction, Input, Output
from bitcoinlib.networks import Network
from random import randint
from time import time
from bitcoinutils.bits import difficulty_to_target, target_to_bits

print ("Server started.")
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('127.0.0.1', 50089)
print('Starting up on {} port {}'.format(*server_address))
server_socket.bind(server_address)
server_socket.listen(10)

def create_work():
    return bytes.fromhex("0100000000000000000000000000000000000000000000000000000000000000000000003BA3EDFD7A7B12B27AC72C3E67768F617FC81BC3888A51323A9FB8AA4B1E5E4A29AB5F49FFFF001D")+bytes.fromhex("00000001")

def difficulty_to_nbits(difficulty):
    # Calculate exponent
    exponent = 0
    while (difficulty // 256**exponent) >= 256:
        exponent += 1

    mantissa = difficulty // 256**exponent

    nbits = (exponent << 24) | mantissa
    return nbits

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

    #Merkle Root comes next, so we need to generate coinbase and bring other txs to verify
    coinbase_tx = Transaction(coinbase=True,network="testnet")
    coinbase_tx.inputs.append(Input(bytes([0 for _ in range(32)]),int(6.25 * 10**8),script=b'Solo Mined by Caleb Markley', sequence=0xffffffff))
    reward_output = Output(6.25 * 10**8, "tb1qwsa28pt43ntyng6zy6adzzhyx03jucvyxf3qja",network="testnet")
    coinbase_tx.outputs.append(reward_output)

    txdat = [coinbase_tx.raw().hex()]
    txids = [coinbase_tx.txid]

    for txid in rpc_connection.getrawmempool()[:9]:
        txids.append(txid)
        txdat.append(rpc_connection.getrawtransaction(txid))

    header = header + merkleCalculator(txids)

    header = header + int(time()).to_bytes(4,"little")

    max_target = 0xffff * 2**(8 * (0x1d - 3))
    target = max_target/rpc_connection.getdifficulty()
    hex_target = hex(int(target))[2:]
    main_part = hex_target[:6]




while False:
    print ("\nWaiting for client to send a message...")
    connection, client_address = server_socket.accept()
    data = connection.recv(1)
    if data[0]==0:
        print ("Client requested work, will be assigned.")
        connection.sendall(create_work())
    if data[0]==1:
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
