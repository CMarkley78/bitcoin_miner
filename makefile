Profiler: Clean
	@mkdir build
	nvcc -o ./build/out.exe ./Profiler/profiler.cu
	./build/out.exe

Library: Clean
	@echo test

Clean:
	@echo Cleaning up environment...
	@if exist build rd /s /q build
	@echo Done!

.PHONY: Profiler Clean Library
