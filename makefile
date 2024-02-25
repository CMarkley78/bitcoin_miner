all: Clean
	@echo default called

Library: Clean
	@echo Building GPU Library...
	@mkdir build
	@nvcc -shared -o ./build/GPU_Miner.dll "./GPU Handler/library.cu" -rdc=false
	@del .\build\GPU_Miner.exp .\build\GPU_Miner.lib
	@echo Done!

Clean:
	@echo Cleaning up environment...
	@if exist build rd /s /q build
	@echo Done!

default: Clean
	@echo Starting miner building/compilation/execution...

.PHONY: Clean Library
