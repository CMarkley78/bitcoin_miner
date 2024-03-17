ifeq ($(OS), Windows_NT)
	rmdir = if exist build rd /s /q
	mkdir = mkdir
else
	rmdir = rm -rf
	mkdir = mkdir -p
endif

BuildLib:
	@echo Building GPU Library...
	@$(mkdir) build
	@nvcc -shared -Xcompiler "-fPIC" -o ./build/GPU_Miner.dll ./GPU_Handler/library.cu -I".\GPU_Handler"
	@echo Done!
HashTest:
	@echo Starting Hashrate Tester... Handing off.
	@python ./HashTest/tester.py
Client:
	@echo Starting client... Handing off.
	@python .\Client\client.py
Server:
	@echo Starting server... Handing off.
	@python .\Server\server.py
clean:
	@echo Resetting build environment...
	@$(rmdir) build
	@echo Done!

.PHONY: clean Client HashTest Server BuildLib
