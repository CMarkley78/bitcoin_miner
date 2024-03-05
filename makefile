Client: Clean
	@echo Building GPU Library...
	@mkdir build
	@nvcc -shared -o ./build/GPU_Miner.dll ./GPU_Handler/library.cu -I".\GPU_Handler"
	@del .\build\GPU_Miner.exp .\build\GPU_Miner.lib
	@echo Done!
	@echo Starting client... Handing off.
	@python .\Client\client.py

ClientNOBUILD: Clean
	@echo Starting client... Handing off.
	@python .\Client\client.py
	
Server: Clean
	@echo Starting server... Handing off.
	@python .\Server\server.py

Clean:
	@cls
	@echo Resetting build environment...
	@if exist build rd /s /q build
	@echo Done!

.PHONY: Clean Client ClientNOBUILD Server
