# Bitcoin Minder
## Storage and distribution of the production version of my Bitcoin Miner. There are several Make targets depending on what should be run. Make automates everything about this repo, and should be the only command that is ever run.

## Keep in mind that no matter what target is being built, it will be done in the "./build" folder. The contents of this folder will be cleared before beginning.

## Make Targets:
### Build_Server
The Build_Server target will build up the Server that is to be connected to by clients in the
