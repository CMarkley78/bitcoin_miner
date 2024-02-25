Library: Clean
	@echo Building GPU Library...
	@mkdir build
	nvcc -o ./build/out.exe "./GPU Handler/library.cu"
	./build/out.exe

Clean:
	@echo Cleaning up environment...
	@if exist build rd /s /q build
	@echo Done!

.PHONY: Profiler Clean Library
