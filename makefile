Profiler:
	if exist build rd /s /q build
	mkdir build
	nvcc -o ./build/out.exe ./Profiler/profiler.cu
	./build/out.exe
	rd /s /q build

.PHONY: Profiler
