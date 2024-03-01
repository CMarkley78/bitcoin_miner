# Bitcoin Miner
## Storage and distribution of the production version of my Bitcoin Miner. There are several Make targets depending on what should be run. Make automates everything about this repo, and should be the only command that is ever run.

## Keep in mind that no matter what target is being built, it will be built in the "./build" folder. The contents of this folder will be cleared before beginning.

## Make Targets:
### Server
The Server target will build up the Server that is to be connected to by clients. There should only be one instance of the server running on one machine. It will generate a file that is to be used as a connection profile. It provides the necessary details required to connect to it.

### Client
The Client target will build up the Client that will connect to a server. Upon being built, it will wait for a connection profile file to be moved into the "./build" folder. Upon this happening, it will make a request for work, and start utilizing the GPU to search possible nonces to solve a block. Keep in mind that the Server that the connection profile details should already be running when the profile is made available to the Client.
