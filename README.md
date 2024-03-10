# Bitcoin Miner

## Keep in mind that no matter what target is being built, it will be built in the "./build" folder. The contents of this folder will be cleared before beginning.

## Make Targets:
### BuildLib
The BuildLib target will build the GPU library so long as the toolchain is installed and in path correctly.
### HashTest
The HashTest target simply times the hasher function on the genesis header, and knows the solution, and will give back the hashrate of the hasher function in MH/s.
### Client
The Client target will just start the client application, and it will assume that the built library is already in the ./build directory.
### Server
The Server target will just start the server application.
### Clean
The Clean target will just remove the ./build directory if it exists.
